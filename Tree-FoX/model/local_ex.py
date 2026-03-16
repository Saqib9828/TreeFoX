# libraries
# ----------------------------------------------------
import json
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import shap

module_dir = os.path.dirname(__file__)
model_file = os.path.join(module_dir, 'trained_model', 'model_20240730_024716')
train_file_url = os.path.join(module_dir, 'dataset', 'train_20240704.csv')
#test_file_url = os.path.join(module_dir, 'dataset', 'test_20240704.csv')
test_file_url = os.path.join(module_dir, 'dataset', 'adversarial_samples.csv')

model = tf.keras.models.load_model(model_file)

def avg_contributor_keys_shap(d):
    result = d.copy()
    key_patterns = ['f_longWord_', 'f_specialKeyword_', 'f_ipaddresses_',
                    'f_sentences_', 'f_ImportsList_create_', 'f_ImportsList_resume_',
                    'f_ImportsList_kill_', 'f_ImportsList_call_', 'f_ImportsList_delete_',
                    'f_ImportsList_other_', 'f_ExportsList_open_', 'f_ExportsList_close_', 'f_ExportsList_create_',
                    'f_ExportsList_call_', 'f_ExportsList_delete_',
                    'f_longWord_', 'f_inValEmails_', 'f_emails_', 'f_DIRs_', 'f_fileName_', 'f_garbage_', 'f_URLs_',
                    'f_ImportsList_open_', 'f_ImportsList_close_', 'f_ExportsList_resume_', 'f_ExportsList_kill_',
                    'f_ExportsList_other_']

    for pattern in key_patterns:
        keys = [key for key in d.keys() if key.startswith(pattern)]
        if len(keys) > 0:
            collected_list = [d[key] for key in keys]
            avg = [sum(x) / len(x) for x in zip(*collected_list)]
            #result[pattern + 'x'] = np.array([avg], dtype=np.float32)
            result[pattern + 'x'] = avg
        for key in keys:
            result.pop(key, None)
    return result

def get_LocalExplanation(train_file_url, test_file_url, model, start_index):
    end_index = start_index + 1
    df_train = pd.read_csv(train_file_url)
    # split target and features
    y_train = df_train['label']
    x_train = df_train.drop(labels=['label', 'filename'], axis=1)

    # encoding
    ohe = OneHotEncoder()
    le = LabelEncoder()
    cols = x_train.columns.values
    for col in cols:
        x_train[col] = le.fit_transform(x_train[col])
    # ohe = OneHotEncoder()
    # x_train = ohe.fit_transform(x_train).toarray()
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    y_train = le.fit_transform(y_train)

    # testing data to  test local ex
    df_test = pd.read_csv(test_file_url)
    # split target and features
    y_test = df_test['label']
    x_test = df_test.drop(labels=['label', 'filename'], axis=1)
    # encoding
    cols = x_test.columns.values
    for col in cols:
        x_test[col] = le.fit_transform(x_test[col])
    sc = StandardScaler()
    x_test = sc.fit_transform(x_test)
    y_test = le.fit_transform(y_test)

    # Use SHAP to explain the predictions of the model
    explainer = shap.Explainer(model.predict, x_train, max_evals=(2 * x_train.shape[1] + 1))

    shap_values = explainer(x_test[start_index:end_index])
    # print(shap_values.values.shape)

    # Get the top n key features for the first class of the first instance
    instance_index = 0
    class_index = 1
    n_top_features = df_test.shape[1]  # for the top, I am selecting all
    class_shap_values = shap_values.values[instance_index, :]
    # print(class_shap_values)
    # key_features = class_shap_values.argsort()[-n_top_features:][::-1]

    # Get the feature names for the key features
    # key_feature_names = df_train.columns[key_features]
    key_feature_names = df_test.columns[:-2]  # removing 'filename', and 'label' column
    # Get the SHAP values for the key features
    # key_feature_values = [class_shap_values[feature] for feature in key_features]
    # Create a dictionary with feature names as keys and SHAP values as values
    # key_features_dict = {key_feature_names[i]: key_feature_values[i] for i in range(len(key_feature_names))}
    key_features_dict = {key_feature_names[i]: [class_shap_values[i][0], class_shap_values[i][1]] for i in
                         range(len(key_feature_names))}
    key_features_dict = avg_contributor_keys_shap(key_features_dict)
    # print("Key features and their values for the first class of the first instance:\n", key_features_dict)
    # Store the dictionary as JSON
    #local_ex_name = "explaination/local_ex_" + str(df_test["filename"][instance_index]) + "_.json"
    #with open(local_ex_name, "w") as fp:
    #    json.dump(key_features_dict, fp)
    file_name = str(df_test["filename"][start_index])
    return key_features_dict, file_name

def get_nLocalExplanation(n):
    model = tf.keras.models.load_model("trained_model/model_20240730_024716")
    train_file_url = 'dataset/train_20240704.csv'
    #test_file_url = 'dataset/test_20240704.csv'
    test_file_url = 'dataset/adversarial_samples.csv'

    result = []
    for i in range(n):
        temp_dict = {}
        start_index = i
        key_features_dict, file_name = get_LocalExplanation(train_file_url, test_file_url, model, start_index)
        temp_dict["file"] = file_name
        temp_dict["data"] = key_features_dict
        result.append(temp_dict)

    return result

def get_LocalExplanationFile(file_name):
    #train_file_url = 'dataset/train.csv'
    #test_file_url = 'dataset/test.csv'
    df_test = pd.read_csv(test_file_url)

    index = df_test.index[df_test['filename'] == file_name][0]
    key_features_dict, file_name = get_LocalExplanation(train_file_url, test_file_url, model, index)

    return key_features_dict

def get_nLabeledLocalExplanation(n):
    df_test = pd.read_csv(test_file_url)

    # Filter out benign and malware files separately
    df_benign = df_test[df_test['label'] == 'benign']
    df_malware = df_test[df_test['label'] == 'malware']

    result_benign = []
    result_malware = []
    benign_files = df_benign.sample(n=n)['filename'].tolist()
    malware_files = df_malware.sample(n=n)['filename'].tolist()

    for file_ in benign_files:
        key_features_dict = get_LocalExplanationFile(file_)
        result_benign.append({'file': file_, 'data': key_features_dict})

    for file_ in malware_files:
        key_features_dict = get_LocalExplanationFile(file_)
        result_malware.append({'file': file_, 'data': key_features_dict})

    return result_benign, result_malware


if __name__ == "__main__":
    #key_features_dict = get_nLocalExplanation(n=3)
    d1, d2 = get_nLabeledLocalExplanation(n=2)
    print(d1)
    print(d2)

print("Completed...!!!")
