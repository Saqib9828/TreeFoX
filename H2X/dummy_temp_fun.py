import os
import pandas as pd
import gzip
from getNumericFeatures import *
from getStringFeatures import *
from ClusterToVec import  *
from getImportExportFeatures import  *

def createPandaInitialTable(featureSize):
    # this portion creating structure of panda table
    file = "setup.exe"
    data = getFeaturesVec(file, file, featureSize)
    columns = []
    for key in data.keys():
        try:
            len_key = len(data[key])
        except:
            len_key = 1
        columns = columns + ['f_' + key + '_' + str(i) for i in range(len_key)]
    columns = columns + ['filename', 'label']
    csv_data = pd.DataFrame(columns=columns)
    return csv_data, columns
    # end section

def appendCSV(csv_data, data, filename, label):
    new_row = []
    for key in data.keys():
        try:
            len_key = len(data[key])
            new_row = new_row + [data[key][i] for i in range(len_key)]
        except:
            len_key = 1
            new_row = new_row + [data[key] for i in range(len_key)]

    new_row = new_row + [filename, label]
    csv_data = pd.concat([csv_data, pd.DataFrame([new_row], columns=columns)])
    return  csv_data

def getFeaturesVec(fname, file, featureSize):
    # return the all features
    numericFeatures = getNumericFeatures(file)
    stringFeatures, importExportList = getStrings(file)
    importExportListFeatures = getImportExportCluster(importExportList)
    dict_combo = {**numericFeatures, **stringFeatures, **importExportListFeatures}
    with open("fun/Features"+str(fname)+".json", "w+") as write_file:
        json.dump(dict_combo, write_file, indent=4)
    stringFeaturesVec = getClusterToVec(stringFeatures, featureSize)
    importExportFeaturesVec = getClusterToVec(importExportListFeatures, featureSize)
    data = dict(list(numericFeatures.items()) + list(stringFeaturesVec.items()) + list(importExportFeaturesVec.items()))
    return data

def processZIP(fname, zipf, featureSize):
    tempf = 'file_'

    f_out = open(tempf, 'wb')
    f_in = gzip.open(zipf, 'rb')
    f_out.writelines(f_in)
    data = getFeaturesVec(fname, tempf, featureSize)

    f_in.close()
    f_out.close()
    os.remove(tempf)
    return data

def check_all_files(csv_data, dir, folder, f_name, avoid_rep=True):
    path = dir + '\\' + folder
    current_files = os.listdir(path)
    no_f = len(current_files)
    countf = 0
    for f in current_files:
        if f_name in f:
            try:
                countf += 1
                data = {}
                zipf = path + '\\' + f
                if f.endswith('.gz'):
                    data = processZIP(f, zipf, featureSize)
                elif "." not in f:
                    data = getFeaturesVec(f, zipf, featureSize)
                else:
                    continue
                csv_data = appendCSV(csv_data, data, f, folder)  # folder name is going to be label
                print(countf, "/", no_f, "done!")
                foutput = "fun/csv_data_output_" + folder + "_" + str(countf) + str(f) + ".csv"
                csv_data.to_csv(foutput, index=False)
                csv_data = csv_data[0:0]
                print("Data copied!")
            except:
                continue
    return csv_data

# convert data to csv data
featureSize = 16
csv_data, columns = createPandaInitialTable(featureSize)


dir = 'E:\saqib_work1\data\miles\\malware_in_class_done'
folder = "dealply"
fname = "origin_4479__"
csv_data = check_all_files(csv_data, dir, folder, fname)
"""
folders = os.listdir(dir)

for folder in folders:
    if os.path.isfile(folder):
        continue
    print("\n", folder)
    print('-----------------------------------------------')
    csv_data = check_all_files(csv_data, dir, folder)
    #foutput = "output/csv_data_output_" + folder + ".csv"
    #csv_data.to_csv(foutput, index=False)
"""
