## matching.py

import os
import re
import json
import math
import numpy as np
import sister
import emb_model


class MatchingADUs():
    def __init__(self, args):
        """
        args : NAME of embedding model (sister or roberta)
        """
        self.args = args


    def cosine_similarity(self, vec1, vec2):

        numerator = sum([vec1[x]*vec2[x] for x in range(len(vec1))])

        sum1 = sum([vec1[x]**2 for x in range(len(vec1))])
        sum2 = sum([vec2[x]**2 for x in range(len(vec2))])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def get_embedding(self, adu):
        """
        Args:
    	adu (str): ADUの内容
        Returns:
    	embedding (list of float): ADUのEmbedding
        """

        if self.args == "roberta":
            self.embedding = emb_model.RobertaEmbedding()
        else:
            self.embedding = sister.MeanEmbedding(lang="en")
            adu = re.sub(r'[,.!?:;"]', '', adu).lower()

        return list(self.embedding(adu))


    def get_matching_pair(self, parent_adu, child_adu, arg_graph):
        """
        Args:
        parent_adu (str): 親となるADUの内容
        child_adu (str): 子となるADUの内容
        arg_graph (dict): マッチング対象のGraph（dictionaryの中身は以下）
        Returns:
    	matching_results (list of matching result): 類似度が高い順番にソートされた結果
    		[
    			(rel_id, from_adu_id, to_adu_id, similarity_score),
    			(rel_id, from_adu_id, to_adu_id, similarity_score),
    			...
    		]
        """
        parent_emb = self.get_embedding(parent_adu)
        child_emb = self.get_embedding(child_adu)

        matching_results = []

        ## arg_graph["relations"] => [dict1, dict2, ...]
        for relation_dict in arg_graph["relations"]:
            ## get 2ADU info
            rel_id = relation_dict["rel_id"]
            from_adu_id = relation_dict["from_adu_id"]
            to_adu_id = relation_dict["to_adu_id"]
            ## set up embeddings from DB
            parent_emb_db = arg_graph["nodes"][from_adu_id]["embedding"]
            child_emb_db = arg_graph["nodes"][to_adu_id]["embedding"]

            ## calcuate similarity
            similarity_parent = self.cosine_similarity(parent_emb, parent_emb_db)
            similarity_child = self.cosine_similarity(child_emb, child_emb_db)

            similarity_score = similarity_parent * similarity_child

            matching_results.append((rel_id, from_adu_id, to_adu_id, similarity_score))

        sort_score = lambda val : val[3]

        return sorted(matching_results, key=sort_score, reverse=True)



"""

json_open = open('./data_Swan/arg_graph_sample.json', 'r')
arg_graphT = json.load(json_open)

Matchclass = MatchingADUs()
results = Matchclass.get_matching_pair("Homework should not be abolished.", "We need to study.", arg_graphT)
print(results)


>>>Loading model...
   [('156', '378', '379', 0.004708969948341021), ('44', '297', '296', 0.003827086123552584), ...

"""
