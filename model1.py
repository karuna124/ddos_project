import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from imblearn.over_sampling import SMOTE

# Set dataset paths
train_file = r'UNSW_NB15_training-set.csv'
test_file = r'UNSW_NB15_testing-set.csv'

# Check files
if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise FileNotFoundError("Training or testing file is missing. Please make sure both are available.")

# Load datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Drop nulls
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Label encode categorical columns using shared encoder where possible
label_encoders = {}
categorical_columns = train_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    combined_values = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined_values)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Show class distribution
print("\nClass distribution in 'attack_cat' (training):")
print(train_df['attack_cat'].value_counts())

# Correlation inspection
correlations = pd.Series(np.corrcoef(train_df.drop('label', axis=1).values.T, train_df['label'].values)[-1][:-1], index=train_df.drop('label', axis=1).columns)
print("\nTop 10 features most correlated with 'label':")
print(correlations.abs().sort_values(ascending=False).head(10))

# Use attack_cat as label
X_train = train_df.drop(['attack_cat'], axis=1)
y_train = train_df['attack_cat']
X_test = test_df.drop(['attack_cat'], axis=1)
y_test = test_df['attack_cat']

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance only training data
y_le = LabelEncoder()
y_train_encoded = y_le.fit_transform(y_train)
y_test_encoded = y_le.transform(y_test)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)

# Random Forest Classifier
rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_resampled, y_train_resampled)
rf_preds = rf.predict(X_test_scaled)

# XGBoost Classifier
xgb = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1, eval_metric='mlogloss')
xgb.fit(X_train_resampled, y_train_resampled)
xgb_preds = xgb.predict(X_test_scaled)

# Accuracy check
print("\nRandom Forest Train Accuracy:", rf.score(X_train_resampled, y_train_resampled))
print("Random Forest Test Accuracy:", rf.score(X_test_scaled, y_test_encoded))
print("\nXGBoost Train Accuracy:", xgb.score(X_train_resampled, y_train_resampled))
print("XGBoost Test Accuracy:", xgb.score(X_test_scaled, y_test_encoded))

# Evaluation
print("\nRandom Forest Report:\n", classification_report(y_test_encoded, rf_preds, zero_division=0))
print("\nXGBoost Report:\n", classification_report(y_test_encoded, xgb_preds, zero_division=0))

# Confusion Matrices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test_encoded, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test_encoded, xgb_preds), annot=True, fmt='d', cmap='Greens')
plt.title('XGBoost Confusion Matrix')
plt.tight_layout()
plt.show()
