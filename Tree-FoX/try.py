import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------------------------
# Load Dataset and Preprocessing
# -----------------------------------------------
dataset = pd.read_csv("model/dataset/merge_csv_samples_20240809.csv")
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)
dataset.loc[dataset.label != 'benign', "label"] = 'malware'

# Remove classes with small counts
value_counts = dataset['label'].value_counts()
to_remove = value_counts[value_counts <= 2000].index
df = dataset[~dataset.label.isin(to_remove)]

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=2)

# Save the training and testing sets as CSV files
train_df.to_csv('model_v2/dataset/train_20240809_v2.csv', index=False)
test_df.to_csv('model_v2/dataset/test_20240809_v2.csv', index=False)

# Load training data
df_train = pd.read_csv('model_v2/dataset/train_20240809_v2.csv')

# -----------------------------------------------
# Feature Extraction and Knowledge Integration
# -----------------------------------------------

# Load domain knowledge weights
with open("model_v2/knowledge_20240809_v2_model_20240809_200313.json", "r") as fp:
    domain_knowledge = json.load(fp)

# Define feature grouping
numerical_features = df_train.columns[:54]  # First 54 features are numerical
complex_features = {
    "f_garbage": [f"f_garbage_{i}" for i in range(16)],
    "f_filename": [f"f_fileName_{i}" for i in range(16)],
    # Add more complex features here as needed
}

# Function to apply domain knowledge weights to features
def apply_weights(feature, feature_values):
    weight = domain_knowledge.get(feature, 1)  # Default weight is 1 if not found
    return feature_values * weight

# Prepare the training data for numerical features
X_train_numeric = []
for index, row in df_train.iterrows():
    numerical_values = row[numerical_features].values
    weighted_numerical_values = [apply_weights(f, numerical_values[i]) for i, f in enumerate(numerical_features)]
    X_train_numeric.append(weighted_numerical_values)

X_train_numeric = np.array(X_train_numeric)
y_train = df_train['label'].apply(lambda x: 1 if x == 'malware' else 0).values  # Convert labels to binary

# Prepare the training data for complex features
X_train_complex = {}
for key, feature_list in complex_features.items():
    X_train_complex[key] = []
    for index, row in df_train.iterrows():
        complex_values = row[feature_list].values
        weighted_complex_values = [apply_weights(f, complex_values[i]) for i, f in enumerate(feature_list)]
        X_train_complex[key].append(weighted_complex_values)
    X_train_complex[key] = np.array(X_train_complex[key])

# -----------------------------------------------
# Model Training
# -----------------------------------------------

# Train model for numerical features
model_numeric = RandomForestClassifier(n_estimators=100, random_state=42)
model_numeric.fit(X_train_numeric, y_train)

# Train models for complex features
models_complex = {}
for key in complex_features.keys():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_complex[key], y_train)
    models_complex[key] = model

# -----------------------------------------------
# Testing and Evaluation
# -----------------------------------------------
# Load test data and apply the same preprocessing steps
df_test = pd.read_csv('model_v2/dataset/test_20240809_v2.csv')

# Prepare the test data for numerical features
X_test_numeric = []
for index, row in df_test.iterrows():
    numerical_values = row[numerical_features].values
    weighted_numerical_values = [apply_weights(f, numerical_values[i]) for i, f in enumerate(numerical_features)]
    X_test_numeric.append(weighted_numerical_values)

X_test_numeric = np.array(X_test_numeric)
y_test = df_test['label'].apply(lambda x: 1 if x == 'malware' else 0).values  # Convert labels to binary

# Prepare the test data for complex features
X_test_complex = {}
for key, feature_list in complex_features.items():
    X_test_complex[key] = []
    for index, row in df_test.iterrows():
        complex_values = row[feature_list].values
        weighted_complex_values = [apply_weights(f, complex_values[i]) for i, f in enumerate(feature_list)]
        X_test_complex[key].append(weighted_complex_values)
    X_test_complex[key] = np.array(X_test_complex[key])

# Predictions and evaluation for numerical features
y_pred_numeric = model_numeric.predict(X_test_numeric)
print(f"Evaluation for Numerical Features:\n{classification_report(y_test, y_pred_numeric)}")

# Predictions and evaluation for complex features
for key in complex_features.keys():
    y_pred_complex = models_complex[key].predict(X_test_complex[key])
    print(f"Evaluation for Complex Features ({key}):\n{classification_report(y_test, y_pred_complex)}")
