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
                                    max_length=self.MAX_SEQ_LEN
                                    )
        else:
            speech_ids = self.tok.encode(speech_tokens,
                                    add_special_tokens=False,
                                    max_length=self.MAX_SEQ_LEN
                                    )

        return speech_ids


    def get_index_multi(self, l, x):
        return [i for i, _x in enumerate(l) if _x == x]


    def get_start_end_index(self, speech_ids, adu_ids):
        """
        input : speech_ids,  adu_ids
        out : start_index, end_index
        """

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

        return start_index, end_index


    def roberta_embedding(self, speech, adu):
        ## get embeddings
        speech_ids = self.roberta_encode(speech, "speech")
        adu_ids = self.roberta_encode(adu, "adu")

        start_index, end_index = self.get_start_end_index(speech_ids, adu_ids)
        assert speech_ids[end_index] == adu_ids[-1], "you got miss match encodeID on adu --> {}".format(adu)

        input_roberta = torch.tensor([speech_ids])

        with torch.no_grad():
            roberta_out = self.model(input_roberta)
        embed_speech, _ = roberta_out

        ## concat (subtraction)
        embed_adu = embed_speech[0][end_index] - embed_speech[0][start_index]

        return embed_adu.numpy().tolist()


    def __call__(self, speech, adu):
        return self.roberta_embedding(speech, adu)
