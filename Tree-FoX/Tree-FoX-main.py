# libraries
# ----------------------------------------------------
import re
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

# -----------------------------------------------
# preprocessing
# -----------------------------------------------

dataset = pd.read_csv("model/dataset/merge_csv_samples_20240809.csv")
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
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
train_df.to_csv('model_v2/dataset/train_20240809_v2.csv', index=False)
test_df.to_csv('model_v2/dataset/test_20240809_v2.csv', index=False)

df_train = pd.read_csv('model_v2/dataset/train_20240809.csv')

