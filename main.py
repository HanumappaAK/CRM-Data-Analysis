import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import KNNImputer

file_path = "/content/Online Retail.xlsx"
data = pd.read_excel(file_path, sheet_name="Online Retail")

# Data Cleaning
print("Initial Dataset Snapshot:")
print(data.head())
print("\nMissing Values Before Cleaning:\n", data.isnull().sum())

# Enhanced data cleaning
data['Customer ID'] = data['CustomerID'].fillna('Unknown')
data[['Quantity', 'Price']] = data[['Quantity', 'UnitPrice']].fillna(data[['Quantity', 'UnitPrice']].median())
for col in ['Country', 'Description']:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing Values After Cleaning:\n", data.isnull().sum())

# Feature Engineering
data['Total_Spend'] = data['Quantity'] * data['Price']
data['Transaction_Frequency'] = data.groupby('Customer ID')['InvoiceNo'].transform('count')
data['Mean_Spend_Per_Transaction'] = data['Total_Spend'] / data['Transaction_Frequency']
data['Items_Per_Transaction'] = data.groupby('InvoiceNo')['Quantity'].transform('sum')
data['Unique_Items'] = data.groupby('Customer ID')['StockCode'].transform('nunique')

# Create a copy of the filtered data
data = data[data['Total_Spend'] > 0].copy()  # Using .copy() to avoid the warning

# Define High Spenders
spend_threshold = data['Total_Spend'].quantile(0.75)
data.loc[:, 'High_Spender'] = (data['Total_Spend'] >= spend_threshold).astype(int)

# Print class distribution
print("\nClass Distribution for High Spenders:\n", data['High_Spender'].value_counts())

# Prepare features for modeling
features = ['Quantity', 'Price', 'Total_Spend', 'Mean_Spend_Per_Transaction', 
           'Transaction_Frequency', 'Items_Per_Transaction', 'Unique_Items']
# Data preprocessing
imputer = KNNImputer(n_neighbors=3)
data_imputed = pd.DataFrame(
    imputer.fit_transform(data[features]), 
    columns=features
)

# Scaling and Normalization
scaler = RobustScaler()
normalizer = QuantileTransformer(output_distribution='uniform')

# Apply transformations
for col in ['Quantity', 'Price', 'Items_Per_Transaction']:
    data_imputed[col] = scaler.fit_transform(data_imputed[[col]])
for col in ['Total_Spend', 'Mean_Spend_Per_Transaction', 'Transaction_Frequency']:
    data_imputed[col] = normalizer.fit_transform(data_imputed[[col]].fillna(0))

# Prepare Training Data
X = data_imputed
y = data['High_Spender']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
dt_model = DecisionTreeClassifier(max_depth=3)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Print Model Performance
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# Visualization 1: Feature Importance
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Customer Spending Prediction')
plt.show()

# Visualization 2: Data Distribution Analysis
plt.figure(figsize=(15, 5))
for i, col in enumerate(['Total_Spend', 'Transaction_Frequency', 'Mean_Spend_Per_Transaction']):
    plt.subplot(1, 3, i+1)
    sns.histplot(data=data, x=col, hue='High_Spender', multiple="stack")
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# Visualization 3: Confusion Matrix with Enhanced Styling
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, xgb_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Spender', 'High Spender'],
            yticklabels=['Low Spender', 'High Spender'])
plt.title("Confusion Matrix - XGBoost Predictions")
plt.show()

# Visualization 4: ROC Curves Comparison
plt.figure(figsize=(8, 6))
# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_xgb, tpr_xgb, color='blue', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

# Decision Tree ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
roc_auc_dt = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_dt, tpr_dt, color='green', label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Visualization 5: Model Performance Metrics Comparison
plt.figure(figsize=(10, 6))
metrics = {
    'XGBoost': {
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1-Score': f1_score(y_test, xgb_pred)
    },
    'Decision Tree': {
        'Accuracy': accuracy_score(y_test, dt_pred),
        'Precision': precision_score(y_test, dt_pred),
        'Recall': recall_score(y_test, dt_pred),
        'F1-Score': f1_score(y_test, dt_pred)
    }
}

df_metrics = pd.DataFrame(metrics)
df_metrics.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Models')
plt.tight_layout()
plt.show()

# Print Key Observations
print("\nKey Observations:")
print("1. Feature Importance Analysis:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.3f}")
print("\n2. Model Performance:")
print(f"   - XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.3f}")
print(f"   - Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.3f}")
print(f"   - ROC AUC Improvement: {(roc_auc_xgb - roc_auc_dt):.3f}")
