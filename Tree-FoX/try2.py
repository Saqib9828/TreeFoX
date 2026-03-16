
import re
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset = pd.read_csv("model/dataset/merge_csv_samples_20240809.csv")
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)

#print(dataset['label'].value_counts()) # print: after data
dataset.loc[dataset.label!='benign',"label"] = 'malware'
print("After merging categories: ", dataset['label'].value_counts())

# Merge categories and remove classes with small counts
print("After merging categories: ", dataset['label'].value_counts())
value_counts = dataset['label'].value_counts()
to_remove = value_counts[value_counts <= 2000].index
df = dataset[~dataset.label.isin(to_remove)]

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=2)
train_df.to_csv('model_v2/dataset/train_20240809_v2.csv', index=False)
test_df.to_csv('model_v2/dataset/test_20240809_v2.csv', index=False)

df_train = pd.read_csv('model_v2/dataset/train_20240809_v2.csv')

# Load features and knowledge
with open('model_v2/features.json', 'r') as f:
    features = json.load(f)

with open('model_v2/knowledge_20240809_v2_model_20240809_200313.json', 'r') as f:
    knowledge_importance = json.load(f)

# Extract numerical and categorical features based on your description
numerical_features = features[:54]  # Index 0 to 53
categorical_features = features[54:]  # The remaining categorical features

# Group categorical features (each group has 16 features)
grouped_categorical_features = {
    "f_garbage": categorical_features[0:16],
    "f_fileName": categorical_features[16:32],
"f_URLs": categorical_features[32:48],
"f_DIRs": categorical_features[48:64],
"f_emails": categorical_features[64:80],
"f_inValEmails": categorical_features[80:96],
"f_longWord": categorical_features[96:112],
"f_specialKeyword": categorical_features[112:128],
"f_ipaddresses": categorical_features[128:144],
"f_sentences": categorical_features[144:160],
"f_ImportsList_open": categorical_features[160:176],
"f_ImportsList_close": categorical_features[176:192],
"f_ImportsList_create": categorical_features[192:208],
"f_ImportsList_resume": categorical_features[208:224],
"f_ImportsList_kill": categorical_features[224:240],
"f_ImportsList_call": categorical_features[240:256],
"f_ImportsList_delete": categorical_features[256:272],
"f_ImportsList_other": categorical_features[272:288],
"f_ExportsList_open": categorical_features[288:304],
"f_ExportsList_close": categorical_features[304:320],
"f_ExportsList_create": categorical_features[320:336],
"f_ExportsList_resume": categorical_features[336:352],
"f_ExportsList_kill": categorical_features[352:368],
"f_ExportsList_call": categorical_features[368:384],
"f_ExportsList_delete": categorical_features[384:400],
"f_ExportsList_other": categorical_features[400:416]
}

# Assign weights to features based on knowledge importance
def assign_weights(feature_list, knowledge_importance):
    weights = []
    for feature in feature_list:
        weights.append(knowledge_importance.get(feature, 1.0))  # Default weight is 1.0 if not found in knowledge
    return weights

numerical_weights = assign_weights(numerical_features, knowledge_importance)
categorical_weights = {group: assign_weights(features, knowledge_importance) for group, features in grouped_categorical_features.items()}

# Prepare the numerical and categorical data for training
X_num_train = train_df[numerical_features].values
X_cat_train = {group: train_df[features].values for group, features in grouped_categorical_features.items()}
y_train = (train_df['label'] == 'malware').astype(int).values

# Apply the weights to the numerical and categorical features
def apply_weights(X, weights):
    # Convert X to a NumPy array, ensuring it's a float array
    X = np.array(X, dtype=float)
    weights = np.array(weights, dtype=float)
    return X * weights


# Apply the weights to the numerical and categorical features
X_num_train_weighted = apply_weights(X_num_train, numerical_weights)
X_cat_train_weighted = {group: apply_weights(X_cat_train[group], categorical_weights[group]) for group in X_cat_train}


# Initialize and train a Random Forest classifier for each feature set
#rf_num = RandomForestClassifier(n_estimators=2, random_state=2)
#rf_cat = {group: RandomForestClassifier(n_estimators=2, random_state=2) for group in grouped_categorical_features}

# Initialize and train a Decision Tree classifier for each feature set
#rf_num = DecisionTreeClassifier(random_state=2)
#rf_cat = {group: DecisionTreeClassifier(random_state=2) for group in grouped_categorical_features}
# Initialize and train a Decision Tree classifier for each feature set with custom parameters
rf_num = DecisionTreeClassifier(
    criterion='entropy',         # Using 'entropy' instead of 'gini' for split quality
    max_depth=5,                # Limiting the depth of the tree to prevent overfitting
    min_samples_split=2,         # Minimum number of samples required to split an internal node
    min_samples_leaf=2,          # Minimum number of samples at a leaf node
    max_features='sqrt',         # Consider the square root of the number of features at each split
    random_state=2               # Ensuring reproducibility
)

# Apply the same parameters for categorical feature groups
rf_cat = {group: DecisionTreeClassifier(
    criterion='entropy',         # Using 'entropy' for all classifiers
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=2
) for group in grouped_categorical_features}

# Train the models
rf_num.fit(X_num_train_weighted, y_train)
for group in X_cat_train_weighted:
    rf_cat[group].fit(X_cat_train_weighted[group], y_train)

# Prepare the test data
X_num_test = test_df[numerical_features].values
X_cat_test = {group: test_df[features].values for group, features in grouped_categorical_features.items()}
y_test = (test_df['label'] == 'malware').astype(int).values

# Apply the weights to the test data
X_num_test_weighted = apply_weights(X_num_test, numerical_weights)
X_cat_test_weighted = {group: apply_weights(X_cat_test[group], categorical_weights[group]) for group in X_cat_test}

# Predict using the trained models for numerical features
if rf_num.n_classes_ == 2:
    y_pred_num = rf_num.predict_proba(X_num_test_weighted)[:, 1]  # Probability of class 1 (malware)
else:
    y_pred_num = rf_num.predict_proba(X_num_test_weighted)[:, 0]  # Probability of the only available class

# Predict using the trained models for categorical features
y_pred_cat = {}
for group in X_cat_test_weighted:
    if rf_cat[group].n_classes_ == 2:
        y_pred_cat[group] = rf_cat[group].predict_proba(X_cat_test_weighted[group])[:, 1]  # Probability of class 1 (malware)
    else:
        y_pred_cat[group] = rf_cat[group].predict_proba(X_cat_test_weighted[group])[:, 0]  # Probability of the only available class

# Aggregate predictions from numerical and categorical models
y_pred_combined = y_pred_num
for group in y_pred_cat:
    y_pred_combined += y_pred_cat[group]  # Simple sum for aggregation

# Thresholding to get final predictions
y_pred_final = (y_pred_combined > len(grouped_categorical_features) / 2).astype(int)  # Threshold at half the number of groups

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_final)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred_final))
