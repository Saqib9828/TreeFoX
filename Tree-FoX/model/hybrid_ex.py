
def get_weightedLocal(weights, local_ex):
    # -----------
    results = {}

    # Loop through the keys in local_ex
    for key in local_ex:
        # Multiply the values in local_ex by the corresponding weight
        results[key] = [val * weights[key] for val in local_ex[key]]

    return results

# Write the results to a new JSON file
# with open('explaination/hybrid_ex.json', 'w') as f:
#    json.dump(results, f)

# extract global important features
def get_globalImportantFeatures(weights, n_top_features):
    # --------------------
    sorted_column_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    # Get the column names with the highest attention weights
    top_columns = [col[0] for col in sorted_column_weights[:n_top_features]]
    return top_columns

def get_hybridImportantFeatures(weights, local_ex, n_top_features):
    # ---------------------
    top_global = get_globalImportantFeatures(weights, n_top_features)
    result = []
    for feature in top_global:
        if(local_ex[feature][0]<local_ex[feature][1]):
            result.append(feature)

    return result

def get_hybridImportantFeatures_for_benign(weights, local_ex, n_top_features):
    # ---------------------
    top_global = get_globalImportantFeatures(weights, n_top_features)
    result = []
    for feature in top_global:
        if(local_ex[feature][0] > local_ex[feature][1]):
            result.append(feature)

    return result


