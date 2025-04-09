# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
sns.set_style('whitegrid')  # Set seaborn style
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import gc

# Create output directory for plots
os.makedirs('output_plots', exist_ok=True)

# --- Load Dataset ---
file_path = "Stroke.csv"  # Updated to match actual file name

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

# Convert scaled data back to DataFrame to preserve feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# --- Model Training ---
# Create multiple models for ensemble
xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=1,
    tree_method='hist',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False
)

# Simplified hyperparameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# LightGBM parameters
lgb_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    n_jobs=1,
    verbose=-1,
    min_data_in_leaf=20,
    min_split_gain=0.0,
    min_child_samples=20
)

# Hyperparameter Tuning for LightGBM
lgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Add memory management before training
gc.collect()

# Perform Grid Search for both models
xgb_grid = GridSearchCV(
    xgb_model, 
    xgb_param_grid, 
    scoring='roc_auc', 
    cv=3,
    n_jobs=1,
    verbose=2
)

lgb_grid = GridSearchCV(
    lgb_model, 
    lgb_param_grid, 
    scoring='roc_auc', 
    cv=3,
    n_jobs=1,
    verbose=2
)

print("Training XGBoost model...")
try:
    # Convert data to float32 to reduce memory usage
    X_train_scaled_df = X_train_scaled_df.astype('float32')
    xgb_grid.fit(X_train_scaled_df, y_train_balanced)
    print("XGBoost training completed successfully!")
except Exception as e:
    print(f"Error in XGBoost training: {str(e)}")
    print("Attempting to continue with LightGBM only...")
    use_xgboost = False
else:
    use_xgboost = True

print("\nTraining LightGBM model...")
try:
    lgb_grid.fit(X_train_scaled_df, y_train_balanced)
    print("LightGBM training completed successfully!")
except Exception as e:
    print(f"Error in LightGBM training: {str(e)}")
    if not use_xgboost:
        print("Both models failed to train. Exiting...")
        exit()

# Create ensemble model with error handling
try:
    if use_xgboost:
        estimators = [
            ('xgb', xgb_grid.best_estimator_),
            ('lgb', lgb_grid.best_estimator_)
        ]
    else:
        estimators = [('lgb', lgb_grid.best_estimator_)]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=1
    )
    
    print("\nTraining ensemble model...")
    ensemble.fit(X_train_scaled_df, y_train_balanced)
    print("Ensemble training completed successfully!")
except Exception as e:
    print(f"Error in ensemble training: {str(e)}")
    exit()

# --- Model Evaluation ---
y_pred = ensemble.predict(X_test_scaled_df)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test_scaled_df)[:, 1])
mcc = matthews_corrcoef(y_test, y_pred)

# Print Model Performance
print("\nModel Performance Metrics:")
print(f"✅ Test Accuracy: {accuracy:.4f}")
print(f"✅ Test Precision: {precision:.4f}")
print(f"✅ Test Recall: {recall:.4f}")
print(f"✅ Test F1-Score: {f1:.4f}")
print(f"✅ Test ROC-AUC: {roc_auc:.4f}")
print(f"✅ Test MCC: {mcc:.4f}")

def save_plot_safely(fig, filename):
    """Helper function to safely save plots"""
    try:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save plot to {filename}: {str(e)}")

# Feature Importance Plot
try:
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': xgb_grid.best_estimator_.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    fig_importance = plt.figure(figsize=(12, 6))
    ax = fig_importance.add_subplot(111)
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Most Important Features')
    save_plot_safely(fig_importance, 'output_plots/feature_importance.png')
except Exception as e:
    print(f"Warning: Could not create feature importance plot: {str(e)}")

# Confusion Matrix Plot
try:
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_matrix = plt.figure(figsize=(8, 6))
    ax = fig_matrix.add_subplot(111)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    save_plot_safely(fig_matrix, 'output_plots/confusion_matrix.png')
except Exception as e:
    print(f"Warning: Could not create confusion matrix plot: {str(e)}")

# Save Models
try:
    joblib.dump(ensemble, 'output_plots/ensemble_model.pkl')
    joblib.dump(scaler, 'output_plots/scaler.pkl')
    print("\n✅ Models saved successfully!")
except Exception as e:
    print(f"Warning: Could not save models: {str(e)}")

# Print file locations
print("\nGenerated files (if successful):")
print("- Feature Importance Plot: output_plots/feature_importance.png")
print("- Confusion Matrix Plot: output_plots/confusion_matrix.png")
print("- Ensemble Model: output_plots/ensemble_model.pkl")
print("- Scaler: output_plots/scaler.pkl")

# Cleanup
plt.close('all')  # Close any remaining figures
