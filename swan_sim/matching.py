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
        args : NAME of embedding model (only "sister" or "roberta" is available.)
        """
        self.args = args
        if self.args == "roberta":
            self.embedding = emb_model.RobertaEmbedding()
        else:
            self.embedding = sister.MeanEmbedding(lang="en")


    def cosine_similarity(self, vec1, vec2):

        numerator = sum([vec1[x]*vec2[x] for x in range(len(vec1))])

        sum1 = sum([vec1[x]**2 for x in range(len(vec1))])
        sum2 = sum([vec2[x]**2 for x in range(len(vec2))])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def get_embedding(self, speech, adu):
        """
        Args:
    	speech (str): speech全体
        adu (str) : ADU内容
        Returns:
    	embedding (list of float): ADUのEmbeddings
        """

        if self.args == "roberta":
            return self.embedding(speech, adu)

        else:
            return list(self.embedding(adu))


    def set_embedding_on_dict(self, speech_dict):
        ## calcuate embedding on arg_graph or new_speech
        for adu_dict in speech_dict["adus"].values():
            speech_id = adu_dict["speech_id"]
            adu = adu_dict["content"]
            speech = speech_dict["speeches"][speech_id]["content"]
            adu_dict["embedding"] = self.get_embedding(speech, adu)

        return speech_dict


    def calcuate_matching_results(self, new_speech, arg_graph):
        """
        Args:
        new_speech (dict): 新しく入力されたスピーチ
        arg_graph (dict): マッチング元のGraph
        Returns:
    	matching_results (list of matching result): 類似度が高い順番にソートされた結果
    		[
    			(adu_id of new_speech, adu_id of arg_graph, similarity_score),
    			(adu_id of new_speech, adu_id of arg_graph, similarity_score),
    			...
    		]
        """

        ## calcuate embedding on speech_dict
        arg_graph = self.set_embedding_on_dict(arg_graph)
        new_speech = self.set_embedding_on_dict(new_speech)

        matching_results = []

        for ns_adu_id in new_speech["adus"]:
            ns_embedding = new_speech["adus"][ns_adu_id]["embedding"]

            for ag_adu_id in arg_graph["adus"]:
                ag_embedding = arg_graph["adus"][ag_adu_id]["embedding"]

                similarity_score = self.cosine_similarity(ns_embedding, ag_embedding)
                matching_results.append((ns_adu_id, ag_adu_id, similarity_score))

        return sorted(matching_results, key=lambda val : val[2], reverse=True)
