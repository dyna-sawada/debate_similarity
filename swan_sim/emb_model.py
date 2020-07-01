## emb_model.py

import numpy
import torch
from transformers import RobertaModel, RobertaTokenizer


class RobertaEmbedding():
    def __init__(self):
        self.tok = RobertaTokenizer.from_pretrained("roberta-base")
        self.special_tokens = ["[Mstart]", "[Mend]"]
        self.tok.add_tokens(self.special_tokens)
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.model.resize_token_embeddings(len(self.tok))
        self.MAX_SEQ_LEN = 256


    def roberta_encode(self, speech):
        # tokenize & encode
        speech_tokens = [self.tok.tokenize(speech)]

        speech_ids = [self.tok.encode(t,
                                    add_special_tokens=True,
                                    max_length=self.MAX_SEQ_LEN,
                                    pad_to_max_length=self.MAX_SEQ_LEN) for t in speech_tokens]

        return speech_ids


    def get_special_tokens_index_list(self, speech_ids):
        Mstart_list = []
        Mend_list = []
        for index, sentence_id in enumerate(encode):
            if sentence_id == self.special_tokens[0]:
                Mstart_list.append(index)
            elif sentence_id == self.special_tokens[1]:
                Mend_list.append(index)

        assert len(Mstart_list) == len(Mend_list), "must be same len."

        return Mstart_list, Mend_list


    def roberta_embedding(self, speech):
        # get embeddings
        speech_ids = self.roberta_encode(speech)
        input_roberta = torch.tensor(input_roberta)

        Msart_list, Mend_list = self.get_special_tokens_index_list(speech_ids)

        with torch.no_grad():
            roberta_out = self.model(input_roberta)
        embed_speech, _ = roberta_out

        embed_adu = torch.stack([embed_speech[0][Mend_list[i]] - embed_speech[0][Msart_list[i]] for i in range(len(s1_list))])

        return embed_adu.numpy()


    def __call__(self, speech):
        return self.roberta_embedding(speech)
