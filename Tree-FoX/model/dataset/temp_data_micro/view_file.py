import json
with open('final_ex_origin_16437__.file.gz.json', 'r') as f:
    feature_file = json.load(f)

print(feature_file["f_ImportsList_other_x"])