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
        ## calcuate cosine_similarity
        numerator = sum([vec1[x]*vec2[x] for x in range(len(vec1))])

        sum1 = sum([vec1[x]**2 for x in range(len(vec1))])
        sum2 = sum([vec2[x]**2 for x in range(len(vec2))])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def get_embedding(self, speech, adu_contents):
        """
        Args:
    	   speech (str): entire of speech
           adu_contents (list of str) : ["adu_1", "adu_2", ...]
        Returns:
    	   embedding (list of float): embeddings of ADU
               [
                   [adu_1 embedding],
                   [adu_2 embedding],
                   ...
               ]
        """

        if self.args == "roberta":
            return self.embedding(speech, adu_contents)

        else:
            adu_embeddings = []
            for adu in adu_contents:
                adu_embedding = list(self.embedding(adu))
                adu_embeddings.append(adu_embedding)
            return adu_embeddings


    def set_embedding_on_dict(self, speech_dict):
        """
        ## calcuate embedding on arg_graph or new_speech
        Args:
            speech_dict (dict): speech dictionary which has some infomationos about adus
        Returns:
            speech_dict (dict): speech dictionary added embedding (key & value) on each adu
        """

        for speech_id in speech_dict["speeches"]:
            adu_ids = []
            adu_contents = []

            if speech_id == "PM":
                continue

            for adu_id in speech_dict["adus"]:
                if speech_dict["adus"][adu_id]["speech_id"] == speech_id:
                    adu_ids.append(adu_id)
                    adu_contents.append(speech_dict["adus"][adu_id]["content"])

            speech = speech_dict["speeches"][speech_id]["content"]
            # calculate embeddings & set on dictionary
            adu_embeddings = self.get_embedding(speech, adu_contents)
            for i in range(len(adu_ids)):
                speech_dict["adus"][adu_ids[i]]["embedding"] = adu_embeddings[i]

        return speech_dict


    def calcuate_matching_results(self, new_speech, arg_graph):
        """
        Args:
            new_speech (dict): new typed speech
            arg_graph (dict): matching source speeches
        Returns:
            matching_results (list of matching result): results sorted by similarity
              [
    			(adu_id of new_speech, adu_id of arg_graph, similarity_score),
    			(adu_id of new_speech, adu_id of arg_graph, similarity_score),
    			...
    		  ]

        if you want greedy one to one matching, use 'align' function.
        """

        ## calcuate embedding on speech_dict
        if "embedding" not in arg_graph["adus"]["108"].keys():
            arg_graph = self.set_embedding_on_dict(arg_graph)
        new_speech = self.set_embedding_on_dict(new_speech)

        matching_results = []

        for ns_adu_id in new_speech["adus"]:
            ns_adu_emb = new_speech["adus"][ns_adu_id]["embedding"]

            for ag_adu_id in arg_graph["adus"]:
                ag_adu_emb = arg_graph["adus"][ag_adu_id]["embedding"]

                similarity_score = self.cosine_similarity(ns_adu_emb, ag_adu_emb)
                matching_results.append((ns_adu_id, ag_adu_id, similarity_score))

        return sorted(matching_results, key=lambda val : val[2], reverse=True)


    def get_most_similar_node(self, ns_adu_id, ns_adu_emb, arg_graph, history):
        sims = []
        aligned_nodes = list(map(lambda x: x[1], history))

        for ag_adu_id in arg_graph["adus"]:
            if ag_adu_id not in aligned_nodes:
                ag_adu_emb = arg_graph["adus"][ag_adu_id]["embedding"]

                similarity_score = self.cosine_similarity(ns_adu_emb, ag_adu_emb)
                sims.append((ns_adu_id, ag_adu_id, similarity_score))

        return max(sims, key=lambda x: x[2])


    def align(self, new_speech, arg_graph):
        """
        Args:
            new_speech (dict): new typed speech
            arg_graph (dict): matching source speeches
        Returns:
            history (list of matching result): only the one with the maximum score for each adu of new_speech
    		  [
    			(adu_1 of new_speech, adu_id of arg_graph, similarity_score,
    			(adu_2 of new_speech, adu_id of arg_graph, similarity_score),
    			...
    		  ]

        if you want every adus's similarity_score, use 'calcuate_matching_results' function.

        """
        history = []

        if "embedding" not in new_speech["adus"]["108"].keys():
            arg_graph = self.set_embedding_on_dict(arg_graph)
        new_speech = self.set_embedding_on_dict(new_speech)

        for ns_adu_id in new_speech["adus"]:
            ns_adu_emb = new_speech["adus"][ns_adu_id]["embedding"]
            msn = self.get_most_similar_node(ns_adu_id, ns_adu_emb, arg_graph, history)
            history += [msn]

        return history
