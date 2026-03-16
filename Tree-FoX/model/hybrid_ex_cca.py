import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA

def get_CORR_cal(weights, shap_values):
    w, v = CCA_cal(weights, shap_values)
    # Compute the dot product between w and v
    dot_product = np.dot(w.reshape(-1), v.reshape(-1))
    #print(f"The correlation between the canonical weights and canonical vectors is: {dot_product}")
    corr, _ = pearsonr(w.reshape(-1), v.reshape(-1))
    return corr

def CCA_cal(weights, shap_values):
    # Convert weights and shap values to numpy arrays
    weights_array = np.array(list(weights.values())).reshape(-1, 1)
    shap_array = np.array(list(shap_values.values()))

    # Standardize the matrices
    weights_array = (weights_array - np.mean(weights_array)) / np.std(weights_array)
    shap_array = (shap_array - np.mean(shap_array, axis=0)) / np.std(shap_array, axis=0)

    # Perform CCA
    cca = CCA(n_components=1)
    cca.fit(weights_array, shap_array)

    # Extract the canonical weights and shap values
    w, v = cca.transform(weights_array, shap_array)

    return w, v

def get_ExplanationMetric(weights, shap_values, n):
    w, v = CCA_cal(weights, shap_values)
    # Sort the features based on their canonical weights
    sorted_features = sorted(list(shap_values.keys()), key=lambda x: -w[list(shap_values.keys()).index(x)][0])

    # Select the top n features
    top_features = {}
    for i in range(n):
        key = sorted_features[i]
        top_features[key] = [w[list(shap_values.keys()).index(key)][0],
                             float(v[list(shap_values.keys()).index(key)][0])]

    return top_features

if __name__ == "__main__":
    # Open the global_ex.json file and load its contents into a dictionary
    with open('explaination/global_ex.json', 'r') as f:
        weights = json.load(f)

    # Open the local_ex.json file and load its contents into a dictionary
    local_ex_file = 'explaination/local_ex_0a3f980be83daad5dc77fc38ffd89fcfe9814a9cc750d04e7fcb4918d9f81ac9_.json'
    with open(local_ex_file, 'r') as f:
        shap_values = json.load(f)

    top_features = get_ExplanationMetric(weights, shap_values, n=3)
    print(top_features)
    print("Completed...!!!")

