# libraries
# ----------------------------------------------------
import re
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
#print(tf.version.VERSION)

from keras import backend as K
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tensorflow.keras.layers import Input, Dense, Dropout, Multiply
import tensorflow.keras.backend as K

import shap
sns.set()
import joblib
# -----------------------------------------------
# preprocessing
# -----------------------------------------------

dataset = pd.read_csv("dataset/merge_csv_samples_20240809.csv")
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
print("Total data length: ", len(dataset))
print("Before duplicate: ", dataset['label'].value_counts()) # print: before data
# dropping ALL duplicate values
#dataset.drop_duplicates(inplace=True)
#print("After dropping duplicate data length: ", len(dataset))
#print("After dropping duplicate: ", dataset['label'].value_counts()) # print: before data
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)

#print(dataset['label'].value_counts()) # print: after data
dataset.loc[dataset.label!='benign',"label"] = 'malware'
print("After merging categories: ", dataset['label'].value_counts())

# to remove class with small no.
value_counts = dataset['label'].value_counts()
to_remove = value_counts[value_counts <= 2000].index
df = dataset[~dataset.label.isin(to_remove)]

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=2)
# Save the training and testing sets as CSV files
train_df.to_csv('dataset/train_20240809_v2.csv', index=False)
test_df.to_csv('dataset/test_20240809_v2.csv', index=False)

df_train = pd.read_csv('dataset/train_20240809.csv')

# split target and features
y_train = df_train['label']
x_train = df_train.drop(labels =['label','filename'],axis=1)

# encoding
ohe = OneHotEncoder()
le = LabelEncoder()
cols = x_train.columns.values
for col in cols:
    x_train[col] = le.fit_transform(x_train[col])
#ohe = OneHotEncoder()
#x_train = ohe.fit_transform(x_train).toarray()
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
y_train = le.fit_transform(y_train)

# create model
# -------------------------------------------------------------------

def create_model():
    inputs = Input(shape=(x_train.shape[1],))

    attention_probs = Dense(1, activation='softmax')(inputs)
    attention_mul = Multiply()([inputs, attention_probs])

    attention_mul = K.reshape(attention_mul, shape=(K.shape(attention_mul)[0], x_train.shape[1]))

    dense1 = Dense(8, activation='relu')(attention_mul)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(8)(dropout1)
    dropout2 = Dropout(0.25)(dense2)
    outputs = Dense(2)(dropout2)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model

# Create a basic model instance
model = create_model()
# Display the model's architecture
# print(model.summary())

# train model
# ----------------------------------------------------------------------------

model.fit(x_train, y_train, epochs=2)

# save model with current date and time
now = datetime.datetime.now()
current_time = now.strftime("%Y%m%d_%H%M%S")
model_name = f"model_{current_time}"
model.save(f"trained_model/{model_name}")

# Save the LabelEncoder
label_encoder_path = f'trained_model/{model_name}/label_encoder.pkl'
joblib.dump(le, label_encoder_path)
# Save the StandardScaler
scaler_path = f'trained_model/{model_name}/scaler.pkl'
joblib.dump(sc, scaler_path)

print("Model trained!!!\------------------------\nModel Name: ", model_name)
print(current_time)





