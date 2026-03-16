# libraries
# ----------------------------------------------------
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# the features have repeated sub-features should to fetch most influencing one contributing for classification
def max_contributor_keys(d):
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
            max_ = np.max([d[key][0] for key in keys])
            result[pattern + 'x'] = np.array([max_], dtype=np.float32)
        for key in keys:
            result.pop(key, None)
    return result

# Extract the attention weights
model = tf.keras.models.load_model("trained_model/model_20240809_200313")
df = pd.read_csv('dataset/train_20240809_v2.csv')
num_columns = df.shape[1]
n_top_features = num_columns # change if you want an specific number of features only
weights = model.layers[1].get_weights()[0]
#print(len(weights))
# Get the column names
column_names = list(df.columns)
# Create a dictionary of column names and their corresponding attention weights
colab_column_weights = dict(zip(column_names, weights))
#colab_column_weights = max_contributor_keys(column_weights)
#colab_column_weights = max_contributor_keys(column_weights)
# Sort the dictionary by attention weights
#sorted_column_weights = sorted(colab_column_weights.items(), key=lambda x: x[1], reverse=True)
# Get the column names with the highest attention weights
# Sort the dictionary by column names
sorted_column_weights = dict(sorted(colab_column_weights.items()))
# Convert sorted dictionary items to a list and slice the top n features
top_columns = [col[0] for col in list(sorted_column_weights.items())[:n_top_features]]

# Convert the list to a dictionary containing only the top columns
top_column_weights = {col: sorted_column_weights[col] for col in top_columns}

# Extract the values from the NumPy arrays (if needed)
# Assuming the weights might be wrapped in a NumPy array
top_column_weights = {k: float(v) if isinstance(v, (list, tuple, np.ndarray)) else v for k, v in top_column_weights.items()}

# Store the dictionary as JSON
with open("knowledge_20240809_v2_model_20240809_200313.json", "w") as fp:
    json.dump(top_column_weights, fp)

print("Completed...!!!")