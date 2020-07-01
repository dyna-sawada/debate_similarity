## main

import json
from matching import MatchingADUs


json_open1 = open('./data_Swan/arg_graph.json', 'r')
json_open2 = open('./data_Swan/new_speech.json', 'r')

arg_graph = json.load(json_open1)
new_speech = json.load(json_open2)

Matchclass = MatchingADUs("roberta")
results = Matchclass.calcuate_matching_results(new_speech, arg_graph)
print(results)
