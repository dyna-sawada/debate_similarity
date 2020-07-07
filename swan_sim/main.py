## main

import os
import json
import pickle
from matching import MatchingADUs


json_open1 = open('./data_Swan/arg_graph.json', 'r')
json_open2 = open('./data_Swan/new_speech.json', 'r')

arg_graph = json.load(json_open1)
new_speech = json.load(json_open2)

mode = "roberta"
m = MatchingADUs(mode)

results = m.align(new_speech, arg_graph)
print(results)



"""
f = open('./data_Swan/Results_RoBERTa.pickle','wb')
pickle.dump(results,f)
f.close
"""
