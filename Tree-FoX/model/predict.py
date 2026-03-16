# libraries
# ----------------------------------------------------
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd

module_dir = os.path.dirname(__file__)
model_file = os.path.join(module_dir, 'trained_model', 'model_20230218_234052')
train_file_url = os.path.join(module_dir, 'dataset', 'train.csv')
#test_file_url = os.path.join(module_dir, 'dataset', 'test.csv')
model = tf.keras.models.load_model(model_file)

def get_trainDF():
    df = pd.read_csv(train_file_url)
    return df

def dataTranformerSeprator(ex_dataset):
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

def predict(x_predict):
    #model = tf.keras.models.load_model("trained_model/model_20230218_234052")
    pred_y = model.predict(x_predict)
    pred_y = np.argmax(pred_y, axis=1)
    return pred_y
