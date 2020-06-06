## main

import os
import re
import math
import numpy as np
import pandas as pd
#import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from KB_data import KnowledgeBaseData
from embeddings import multiplication_sims




def main():
    LO_span_texts_lists = []
    KB_datasets = []
    with zipfile.ZipFile('./data_Swan/export_ArgGraph_.zip', 'r') as zip_data:
        filename_list = zip_data.namelist()

        for filename in filename_list:
            m = re.match('ArgGraph__(.*)_(.*).graph@swan.jp.xml', filename)
            if not m:
                continue
            if m.group(2) == "ano2":
                continue       ## pick up only ano1_data

            debate_id = m.group(1)
            df = pd.read_csv('./data_Swan/original_data.tsv', delimiter='\t')
            original_text = (df[df['name'] == debate_id])[:1]['text'].values[0]
            xml = zip_data.read(filename)

            KBClass = KnowledgeBaseData(xml, original_text)
            _, LO_span_texts_list = KBClass.get_span_text_info()
            LO_span_texts_lists += LO_span_texts_list
            KB_data = KBClass.KB_data()
            KB_datasets += KB_data

        word_freqs = Counter(LO_span_texts_lists)

        ## load pretrained embedding model
        #PATH_TO_GLOVE = "./data_model/glove.840B.300d.txt"
        #glove2word2vec(PATH_TO_GLOVE, tmp_file)
        tmp_file = "./data_model/glove.840B.300d.w2v.txt"
        glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

        ## 今はinput_dataを固定してます．
        input_data = ["if you do homework efficiently, you will have more free time",
              "There is no reason homework should be abolished"]

        value, index = multiplication_sims(input_data, datasets, glove, word_freqs)
        print(datasets[index])


if __name__ == "__main__":
    main()


# >> ['what we learn at school is very significant and we often apply that knowledge to our favorite things to do',
 #'the most important work for students is study',
 #'support',
 #'生徒の一番の仕事は勉強、とスタンスをクリアにしたことがよかったです']
