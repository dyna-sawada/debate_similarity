## embeddings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import math



def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def calculate_similarities_sif(input_text, KB_texts, model, freqs={}, a=0.001):
    """
    # SIF model
    << in >>
    input_text : [text]
    a : parameter
    << out >>
    similarities : [sim1, sim2, ...]
    """
    total_freq = sum(freqs.values())

    embeddings = []

    ## SIF requires us to first collect all sentence embeddings and then perform
    ## common component analysis.
    for KB_text in KB_texts:
        input_tokens = [token for token in input_text if token in model]
        KB_tokens = [token for token in KB_text if token in model]

        weights1 = [a/(a+freqs.get(token,0)/total_freq) for token in input_tokens]
        weights2 = [a/(a+freqs.get(token,0)/total_freq) for token in KB_tokens]

        embedding1 = np.average([model[token] for token in input_tokens], axis=0, weights=weights1)
        embedding2 = np.average([model[token] for token in KB_tokens], axis=0, weights=weights2)

        embeddings.append(embedding1)
        embeddings.append(embedding2)

    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx*2].reshape(1, -1),
                              embeddings[idx*2+1].reshape(1, -1))[0][0]
            for idx in range(int(len(embeddings)/2))]

    return np.array(sims)


def multiplication_sims(input_data, KB_data, model, freqs):
    """
    << in >>
    input_data : [text1, text2]
    << out >>
    1. maximum value of calcuated similarities
    2. index of maximum value
    """
    ## set new data
    input_1 = re.sub(r'[,.!?:;"]', '', input_data[0]).lower().split()
    input_2 = re.sub(r'[,.!?:;"]', '', input_data[1]).lower().split()
    ## set KB data
    from_text_data_list = [data_info[0].split() for data_info in KB_data]
    to_text_data_list = [data_info[1].split() for data_info in KB_data]
    ## calculate
    from_sims = calculate_similarities_sif(input_1, from_text_data_list, model, freqs)
    to_sims = calculate_similarities_sif(input_2, to_text_data_list, model, freqs)

    multi_sims = from_sims * to_sims

    return np.max(multi_sims), np.argmax(multi_sims)
