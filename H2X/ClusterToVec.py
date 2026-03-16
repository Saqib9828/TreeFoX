import json
import re
import math
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
maxWordLen = 5

def convertSent(s):
    sent = ""
    word = ""
    for i in range(len(s)-1):
        # if word is null, asign whatever current letter
        if word == "":
            word = word + s[i]
            continue
        # if no change found
        if (s[i].islower() and s[i+1].islower()) or (s[i].isupper() and s[i+1].isupper()):
            #keep adding letter in word
            word = word + s[i]
        else:
            # if change found
            # 1. lower followed by upper
            if s[i].islower() and s[i+1].isupper():
                word = word + s[i]
                sent = sent + word + " "
                word = ""
            #2. upper folled by lower
            elif s[i].isupper() and s[i+1].islower():
                sent = sent + word + " "
                word = s[i]
    # handel last letter
    word =  word + s[-1]
    sent = sent + word
    return sent

def processData(s):
    s = re.sub('[^a-zA-Z0-9 \n]', ' ', s)
    s = s.split(' ')
    s = list(set(s))
    str_ = ""
    for i in range(len(s)):
        if len(s[i]) > maxWordLen:
            s[i] = convertSent(s[i])
        str_ = str_ + " " + s[i]
    str_ = " ".join(str_.split())
    return str_

def getClusterEmbedding(s):
    x = list(map(processData, s))
    sentence_embeddings = sbert_model.encode(x)
    col_totals = [ sum(x) for x in zip(*sentence_embeddings) ] #sum
    return [x / len(s) for x in col_totals] #average

def getPooling(vector, featureSize):
    window_size = math.ceil(len(vector)/featureSize)
    f = [0 for i in range(featureSize)]
    for i in range(featureSize):
        sum_ = 0
        for j in range(window_size):
            if window_size*i+j >= len(vector):
                break
            sum_ = sum_ + vector[window_size*i+j]
        f[i] = sum_
    return f

def getClusterToVec(data, featureSize):
    res = {}
    for key in data.keys():
        x = getClusterEmbedding(data[key])
        x = getPooling(x, featureSize)
        res[key] = x
    return res
def getClusterToVec_mixData(data, featureSize):
    res = {}
    for key in data.keys():
        if isinstance(data[key], list):
            x = getClusterEmbedding(data[key])
            x = getPooling(x, featureSize)
            res[key] = x
            #res[key] = [0 for i in range(16)]
        else:
            res[key] = data[key]
    return res
'''
f = open('Strings.json')
data = json.load(f)
featureSize = 4
res = getClusterToVec(data, featureSize)
print(json.dumps(res, indent=4))
'''