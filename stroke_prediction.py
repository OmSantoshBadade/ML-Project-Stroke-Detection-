# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# --- Load Dataset ---
file_path = "your_stroke_data.csv"  # Replace with your file path

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!\n")
else:
    print(f"❌ Error: File '{file_path}' not found. Exiting script.")
    exit()

# Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# --- Data Preprocessing ---
# Identify the correct target column
target_column_name = "Family History of Stroke"  # Replace with the exact column name

if target_column_name not in data.columns:
    print(f"❌ Error: '{target_column_name}' column not found in dataset!")
    print("Available columns:", list(data.columns))
    exit()

print(f"✅ Target column identified: {target_column_name}")

# Handle missing values
# For numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# For categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
else:
    print("⚠️ No categorical columns found for missing value imputation.")

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# --- Feature Engineering ---
features = data.drop(columns=[target_column_name])
target = data[target_column_name]

# Check for class imbalance
print("\nClass Distribution:")
print(target.value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train Model Using Logistic Regression ---
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = logreg_model.predict(X_test)
y_pred_proba = logreg_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print Model Performance
print("\n✅ Model Performance:")
print(f"✅ Test Accuracy: {accuracy:.4f}")
print(f"✅ Test Precision: {precision:.4f}")
print(f"✅ Test Recall: {recall:.4f}")
print(f"✅ Test F1-Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc_auc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance (Coefficients)
coefficients = pd.DataFrame({
    "Feature": features.columns,
    "Coefficient": logreg_model.coef_[0]
})
print("\nFeature Coefficients (Importance):")
print(coefficients.sort_values(by="Coefficient", ascending=False))

# Save Model and Scaler
joblib.dump(logreg_model, 'logreg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model and scaler saved successfully as 'logreg_model.pkl' and 'scaler.pkl'.")