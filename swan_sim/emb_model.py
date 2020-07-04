## emb_model.py

import numpy
import torch
from transformers import RobertaModel, RobertaTokenizer


class RobertaEmbedding():
    def __init__(self):
        self.tok = RobertaTokenizer.from_pretrained("roberta-base")

        self.model = RobertaModel.from_pretrained("roberta-base")

        self.MAX_SEQ_LEN = 256


    def roberta_encode(self, speech, string):
        ## tokenize & encode
        speech_tokens = self.tok.tokenize(speech)

        if string == "speech":
            speech_ids = self.tok.encode(speech_tokens,
                                    add_special_tokens=True,
                                    max_length=self.MAX_SEQ_LEN,
                                    )
        else:
            speech_ids = self.tok.encode(speech_tokens)

        return speech_ids


    def get_index_multi(self, l, x):
        return [i for i, _x in enumerate(l) if _x == x]


    def get_start_end_index(self, speech_ids, adu_ids_list):
        """
        Args:
            speech_ids (list of int): encode of entire speech
            adu_ids_list (list of int): encode of every adu on speech
        Returns:
            start_index_list (list of int): start_index of every adu on encode of entire speech
            end_index_list (list of int): end_index of every adu on encode of entire speech
        """

        start_index_list = []
        end_index_list = []

        for adu_ids in adu_ids_list:
            index_front = []
            index_back = []

            for i, adu_id in enumerate(adu_ids[1:]):
                where_id = 0

                index_on_sp_ids_list = self.get_index_multi(speech_ids, adu_id)
                if len(index_on_sp_ids_list) == 1:
                    index_on_adu_ids = i
                    break

                index_front = index_on_sp_ids_list

                ## whether 2 tokens are continuous
                count = 0
                for f in index_front:
                    for b in index_back:
                        if b > f:
                            continue
                        if b + 1 == f:
                            count += 1
                            where_id = index_front.index(f)

                if count == 1:
                    index_on_adu_ids = i
                    break

                index_back = index_on_sp_ids_list


            start_index = index_on_sp_ids_list[where_id] - (index_on_adu_ids + 1)
            end_index = index_on_sp_ids_list[where_id] + (len(adu_ids[1:]) - index_on_adu_ids - 1)
            start_index_list.append(start_index)
            end_index_list.append(end_index)

        return start_index_list, end_index_list


    def roberta_embedding(self, speech, adu_contents):
        ## get embeddings
        speech_ids = self.roberta_encode(speech, "speech")

        adu_ids_list = []
        for adu_content in adu_contents:
            adu_ids = self.roberta_encode(adu_content, "adu")
            adu_ids_list.append(adu_ids)

        start_index_list, end_index_list = self.get_start_end_index(speech_ids, adu_ids_list)

        input_roberta = torch.tensor([speech_ids])

        with torch.no_grad():
            roberta_out = self.model(input_roberta)
        embed_speech, _ = roberta_out

        embed_adus = []
        for i in range(len(start_index_list)):
            ## concat (subtraction)
            embed_adu = embed_speech[0][end_index_list[i]] - embed_speech[0][start_index_list[i]]
            embed_adu.numpy().tolist()
            embed_adus.append(embed_adu)

        return embed_adus


    def __call__(self, speech, adu_contents):
        return self.roberta_embedding(speech, adu_contents)
