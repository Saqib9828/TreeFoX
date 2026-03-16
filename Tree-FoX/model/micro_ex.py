import json
import numpy as np
import pandas as pd
import gzip
from getNumericFeatures import *
from getStringFeatures import *
from ClusterToVec import  *
from getImportExportFeatures import  *
from utility import *
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
"""
def get_importance_intra_feature(test_x, pre_y, i, m, min_len, model):
    if pre_y[i] == 1:
        # select a random instance with pre_y == 0
        j = np.random.choice([index for index, label in enumerate(pre_y) if label == 0])
        # create a copy of the instance with feature m replaced by the corresponding feature in the random instance
        updated_instance = test_x[i].copy()
        updated_instance[m] = test_x[j][m]
        # divide the feature m into two parts
        half_len = len(updated_instance[m]) // 2
        a = updated_instance[m][:half_len]
        b = updated_instance[m][half_len:]
        # create two new instances with either part of the feature m
        updated_instance_a = updated_instance.copy()
        updated_instance_a[m] = a
        updated_instance_b = updated_instance.copy()
        updated_instance_b[m] = b
        # predict the label for both instances
        pred_a = model.predict(np.array([updated_instance_a]))
        pred_b = model.predict(np.array([updated_instance_b]))
        # if the prediction is still 1, repeat the process with half of the feature m
        if pred_a == 1 or pred_b == 1:
            get_importance_intra_feature(test_x, pre_y, i, m, min_len, model)
        # if the prediction becomes 0, the feature m is considered as a canonical feature
        else:
            if len(updated_instance[m]) > min_len:
                get_importance_intra_feature(test_x, pre_y, i, m, min_len, model)
"""
def split_dict(dct, key_):
    length = len(dct[key_])
    dict1 = {}
    dict2 = {}

    for key, value in dct.items():
        if key == key_:
            dict1[key] = value[:length // 2]
            dict2[key] = value[length // 2:]
        else:
            dict1[key] = dict2[key] = value

    return dict1, dict2

def dataTranformerSeprator_df(ex_dataset):
    ex_y_test = ex_dataset['label']
    ex_x_test = ex_dataset.drop(labels=['label', 'filename'], axis=1)
    ohe = OneHotEncoder()
    le = LabelEncoder()
    cols = ex_x_test.columns.values
    for col in cols:
        ex_x_test[col] = le.fit_transform(ex_x_test[col])
    ex_y_test = le.fit_transform(ex_y_test)
    # ohe = OneHotEncoder()
    # x_train = ohe.fit_transform(x_train).toarray()
    sc = StandardScaler()
    ex_x_test = sc.fit_transform(ex_x_test)
    return ex_x_test, ex_y_test
"""
def dataTranformerSeprator(ex_dataset):
    ex_y_test = ex_dataset['label']
    ex_dataset = ex_dataset.to_frame()
    print(ex_dataset)
    ex_x_test = ex_dataset.drop(labels=['label', 'filename'], axis=1)
    ohe = OneHotEncoder()
    le = LabelEncoder()
    cols = ex_x_test.columns.values
    for col in cols:
        ex_x_test[col] = le.fit_transform(ex_x_test[col])
    ex_y_test = le.fit_transform(ex_y_test)
    # ohe = OneHotEncoder()
    # x_train = ohe.fit_transform(x_train).toarray()
    sc = StandardScaler()
    ex_x_test = sc.fit_transform(ex_x_test)
    return ex_x_test, ex_y_test
"""
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def generate_column_names(featureSize, Key):
    column_names = []
    for i in range(featureSize):
        column_names.append(f"f_{Key}_{i}")
    return column_names

def getPredictionFor_updateFeature(feature_file, df_test, j, model, key, min_len, featureSize):
    data = getClusterToVec_mixData(feature_file, featureSize)
    csv_data, columns = createPandaInitialTable(featureSize)
    ex_data = appendCSV(csv_data, data, "temp_", "malware", columns)
    #ex_x_test, ex_y_test = dataTranformerSeprator(ex_data)
    # create a copy of the instance with feature m replaced by the corresponding feature in the random instance
    updated_instance = df_test.iloc[[j]].copy()
    column_names = generate_column_names(featureSize, key)
    #updated_instance_x[column_names] = ex_x_test[column_names]
    updated_instance[column_names] = ex_data.iloc[0][column_names]
    updated_instance_x, updated_instance_y = dataTranformerSeprator_df(updated_instance)
    updated_instance_pred_y = model.predict(updated_instance_x)
    updated_instance_pred_y = np.argmax(updated_instance_pred_y, axis=1)
    return updated_instance_pred_y

def getImportantArgumnetInFeature_m(feature_file, df_test, j, model, key, min_len, featureSize):
    imp_features_list = []
    if len(feature_file[key]) <= min_len:
        imp_features_list.append(feature_file[key])
    else:
        feature_file1, feature_file2 = split_dict(feature_file, key)
        feature_file1_pred_y = getPredictionFor_updateFeature(feature_file1, df_test, j, model, key, min_len, featureSize)
        feature_file2_pred_y = getPredictionFor_updateFeature(feature_file2, df_test, j, model, key, min_len,
                                                              featureSize)
        #feature_file1_pred_y = 0
        # if 1st part got malicious -> split further
        if (feature_file1_pred_y == 1):
            imp_features_list1 = getImportantArgumnetInFeature_m(feature_file1, df_test, j, model, key, min_len, featureSize)
            imp_features_list.append(imp_features_list1)

        # if 2nd part got malicious -> split further
        if (feature_file2_pred_y == 1):
            imp_features_list2 = getImportantArgumnetInFeature_m(feature_file2, df_test, j, model, key, min_len, featureSize)
            imp_features_list.append(imp_features_list2)

    return imp_features_list


def microExExtractor(feature_file, df_test, j, model, key, min_len, featureSize):
    micro_ex = {} # store key and their values in a dict

    return micro_ex
# main
featureSize = 16
min_len = 10
model = tf.keras.models.load_model("trained_model/model_20230218_234052")
df_test = pd.read_csv('dataset/test.csv')
x_test, y_test = dataTranformerSeprator_df(df_test)
# select a random instance with pre_y == 0
j = np.random.choice([index for index, label in enumerate(y_test) if label == 0])
with open('dataset/temp_data_micro/Ex_Features_origin_16437__.file.gz.json', 'r') as f:
    feature_file = json.load(f)

#data = getClusterToVec_mixData(feature_file, featureSize)
#csv_data, columns = createPandaInitialTable(featureSize)
#csv_data = appendCSV(csv_data, data, "temp_", "temp_")
key = 'sentences'
result = flatten_list(getImportantArgumnetInFeature_m(feature_file, df_test, j, model, key, min_len, featureSize))
#micro_ex = microExExtractor(csv_data, model, min_len, featureSize)
print(result)
print("Completed...!!!")