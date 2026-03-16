# libraries
# ----------------------------------------------------
import re
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from keras import backend as K
import matplotlib.pyplot as plt
import itertools

from tensorflow.keras.layers import Input, Dense, Dropout, Multiply
import tensorflow.keras.backend as K

import shap

sns.set()

df_test = pd.read_csv('dataset/test_20240809_v2.csv')

# split target and features
y_test = df_test['label']
x_test = df_test.drop(labels=['label', 'filename'], axis=1)

# encoding
ohe = OneHotEncoder()
le = LabelEncoder()
cols = x_test.columns.values
for col in cols:
    x_test[col] = le.fit_transform(x_test[col])

sc = StandardScaler()
x_test = sc.fit_transform(x_test)
y_test = le.fit_transform(y_test)

model = tf.keras.models.load_model("trained_model/model_20240809_200313")

# Throughput calculation function
def calculate_throughput(x_test, y_test, num_files):
    x_test_sample = x_test[:num_files]
    y_test_sample = y_test[:num_files]

    t1 = datetime.datetime.now()
    y_pred = np.argmax(model.predict(x_test_sample), axis=-1)
    t2 = datetime.datetime.now()

    # Calculate time taken and throughput
    time_taken = (t2 - t1).total_seconds()
    throughput = num_files / time_taken if time_taken > 0 else float('inf')

    return throughput, time_taken

# Throughput for different file sizes and saving results
file_sizes = list(range(1, 5))  # File sizes from 1 to 1000
throughput_data = []

for num_files in file_sizes:
    throughput, time_taken = calculate_throughput(x_test, y_test, num_files)
    avg_time_per_file = time_taken / num_files if num_files > 0 else 0
    throughput_data.append([num_files, time_taken, avg_time_per_file])
    print(f"Throughput for {num_files} files: {throughput:.2f} files/sec (Time taken: {time_taken:.4f} sec)")

# Convert to DataFrame and save as CSV
throughput_df = pd.DataFrame(throughput_data, columns=['file_size', 'total_time_taken', 'average_time_taken'])
throughput_df.to_csv('trained_model/model_20240809_200313/throughput_results_v_faltu.csv', index=False)

# Evaluating on the full dataset
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# Predict on the full dataset
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Create confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_mat)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
