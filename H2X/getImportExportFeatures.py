import  json
def getKeyWord(data, key):
    res = []
    for item in data:
        if key in item.lower():
            res.append(item)
    return res

def getImportExportCluster(data):
    keywords = ['open', 'close', 'create', 'resume', 'kill', 'call', 'delete']
    res = {}
    for key in data.keys():
         temp = []
         for word in keywords:
             res[key + '_' + word] = getKeyWord(data[key], word)
             temp = temp + res[key + '_' + word]
         res[key + '_other'] = list(set(data[key]) - set(temp))
    return res
'''
importExport = {
    'a' :[
        'saqOpen', 'rat', 'opKill'
    ]
    ,
    'b' :[
        'saq', 'op'
    ]
}
importExport = getImportExportCluster(importExport)
print(json.dumps(importExport, indent=4))
'''










