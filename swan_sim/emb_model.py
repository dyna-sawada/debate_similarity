## emb_model.py

import numpy
import torch
from transformers import RobertaModel, RobertaTokenizer


class RobertaEmbedding():
    def __init__(self):
        self.tok = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.MAX_SEQ_LEN = 256


    def roberta_encode(self, speech):
        # tokenize & encode
        speech_tokens = [self.tok.tokenize(speech)]

        speech_ids = [self.tok.encode(t,
                                    add_special_tokens=True,
                                    max_length=self.MAX_SEQ_LEN,
                                    pad_to_max_length=self.MAX_SEQ_LEN) for t in speech_tokens]

        return torch.tensor(speech_ids)


    def roberta_embedding(self, speech):
        # get embeddings
        input_roberta = self.roberta_encode(speech)

        with torch.no_grad():
            roberta_out = self.model(input_roberta)
        embed_speech = roberta_out[1]
        embed_speech_np = embed_speech.numpy()

        return embed_speech_np[0]


    def __call__(self, speech):
        return self.roberta_embedding(speech)
