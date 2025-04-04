# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# --- Load Dataset ---
file_path = "stroke_data.csv"  # Updated to match actual file name

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!\n")
else:
    print(f"❌ Error: File '{file_path}' not found. Exiting script.")
    exit()

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# --- Data Preprocessing ---
# Identify the target column (assuming it's named 'stroke' or similar)
target_column_name = None
possible_target_names = ['stroke', 'Stroke', 'STROKE', 'Family History of Stroke']
for col in data.columns:
    if any(name in col for name in possible_target_names):
        target_column_name = col
        break

if target_column_name is None:
    print("❌ Error: Target column not found in dataset!")
    print("Available columns:", list(data.columns))
    exit()

print(f"✅ Target column identified: {target_column_name}")

# Handle missing values more intelligently
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill numeric missing values with median
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill categorical missing values with mode
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# --- Feature Engineering ---
features = data.drop(columns=[target_column_name])
target = data[target_column_name]

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale features using RobustScaler (more robust to outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
# Create multiple models for ensemble
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

lgb_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    n_jobs=-1
)

# Hyperparameter Tuning for XGBoost
xgb_param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Hyperparameter Tuning for LightGBM
lgb_param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'feature_fraction': [0.8, 0.9, 1.0],
    'bagging_fraction': [0.8, 0.9, 1.0]
}

# Perform Grid Search for both models
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
lgb_grid = GridSearchCV(lgb_model, lgb_param_grid, scoring='roc_auc', cv=5, n_jobs=-1)

print("Training XGBoost model...")
xgb_grid.fit(X_train_scaled, y_train_balanced)
print("Training LightGBM model...")
lgb_grid.fit(X_train_scaled, y_train_balanced)

# Create ensemble model
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_grid.best_estimator_),
        ('lgb', lgb_grid.best_estimator_)
    ],
    voting='soft'
)

# Train ensemble model
print("Training ensemble model...")
ensemble.fit(X_train_scaled, y_train_balanced)

# --- Model Evaluation ---
y_pred = ensemble.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test_scaled)[:, 1])
mcc = matthews_corrcoef(y_test, y_pred)

# Print Model Performance
print("\nModel Performance Metrics:")
print(f"✅ Test Accuracy: {accuracy:.4f}")
print(f"✅ Test Precision: {precision:.4f}")
print(f"✅ Test Recall: {recall:.4f}")
print(f"✅ Test F1-Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc_auc:.4f}")
print(f"✅ Test MCC: {mcc:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': xgb_grid.best_estimator_.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save Models
joblib.dump(ensemble, 'ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Models saved successfully!")
