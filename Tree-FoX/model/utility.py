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
    data = getFeaturesVec(file, featureSize)
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

def appendCSV(csv_data, data, filename, label, columns):
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

def getFeaturesVec(file, featureSize):
    # return the all features
    numericFeatures = getNumericFeatures(file)
    stringFeatures, importExportList = getStrings(file)
    importExportListFeatures = getImportExportCluster(importExportList)
    stringFeaturesVec = getClusterToVec(stringFeatures, featureSize)
    importExportFeaturesVec = getClusterToVec(importExportListFeatures, featureSize)
    data = dict(list(numericFeatures.items()) + list(stringFeaturesVec.items()) + list(importExportFeaturesVec.items()))
    return data

def processZIP(zipf, featureSize):
    tempf = 'file_'

    f_out = open(tempf, 'wb')
    f_in = gzip.open(zipf, 'rb')
    f_out.writelines(f_in)
    data = getFeaturesVec(tempf, featureSize)

    f_in.close()
    f_out.close()
    os.remove(tempf)
    return data




