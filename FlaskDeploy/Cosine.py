import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time

SBERT = SentenceTransformer('all-MiniLM-L6-v2')
corpus_df = pd.read_csv("D:\\chatbot1212\\Model\\Data\\STSB\\sentence_list.csv", header=0)
corpus_embeddings = SBERT.encode(corpus_df['sentence'], convert_to_tensor=True)


def get_index_sbert(query, group, top_k, s):
    s.sendall(bytes('clr-bm', "utf8"))
    query_embedding = SBERT.encode(query, convert_to_tensor=True)
    result_df = pd.read_csv(
        "D:\\chatbot1212\\Model\\Data\\STSB\\sentence_list.csv",
        header=0)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    result_list = cos_scores
    for similarity in result_list:
        time.sleep(0.05)
        s.sendall(bytes('bm_' + str(format(similarity * 10, '.1f')), "utf8"))
    result_df['similarity'] = cos_scores
    mean_df = result_df.groupby(["label_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max_list = []
    max_sentence_list = []
    try:
        max_list = mean_df.iloc[-top_k:]['label_index'].tolist()
        max_list.reverse()
        for max_id in max_list:
            group_df = result_df[result_df['label_index'] == max_id]
            idx = group_df['similarity'].idxmax()
            max_sentence_list.append(result_df.iloc[idx]['sentence'][:300])
    except ValueError:
        index = -1
    return max_list, max_sentence_list
