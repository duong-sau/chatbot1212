# group classification
import time
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from FlaskDeploy.BM25 import word_token
from Model.Common import get_similarity


def get_cluster(input_query, top_p):
    result_df = pd.read_csv("..\\Data\\IntentClassification\\sentence_list.csv", header=0)
    input_corpus = result_df['sentence'].tolist()
    docs = [input_query] + input_corpus
    docs = [word_token(d, lemma=True) for d in docs]
    tokenized_corpus = [doc.split(' ') for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus[1:])
    input_query = tokenized_corpus[0]
    result_df['similarity'] = bm25.get_scores(input_query)
    mean_df = result_df.groupby(["cluster_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max_list = []
    try:
        max_list = mean_df.iloc[-top_p:]['cluster_index'].tolist()
        max_list.reverse()
        for max_id in max_list:
            group_df = result_df[result_df['cluster_index'] == max_id]
            idx = group_df['similarity'].idxmax()
    except ValueError:
        index = -1
    return max_list


# answer the question
def get_index(question, group, top_k, siamese_tokenizer, siamese_model):
    result_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/labelClassification/sentence_list"
        ".csv",
        header=0)
    result_df = result_df[result_df['cluster_index'].isin(group)]
    for i, r in tqdm(result_df.iterrows(), total=len(result_df)):
        compare_sentences = r["sentence"]
        similarity = get_similarity(tokenizer=siamese_tokenizer, model=siamese_model, test_sentence=question,
                                    compare_sentences=compare_sentences)
        result_df.loc[i, "similarity"] = similarity
    result_df['similarity'] = pd.to_numeric(result_df['similarity'], errors='coerce')
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


def pandas_to_json(answer_df):
    js = answer_df.to_json(orient='index')
    return js


def estimate_time(siamese_model, siamese_tokenizer):
    start = time.time()
    question = "In the partition model, you can specify a substitution model for each gene/character set. IQ-TREE " \
               "will then estimate the model parameters separately for every partition. Moreover, IQ-TREE provides " \
               "edge-linked or edge-unlinked branch lengths between partitions. "
    compare_sentences = "That means, part1 contains sites 1-100 and 200-384 of the alignment. Another example is"
    similarity = get_similarity(tokenizer=siamese_tokenizer, model=siamese_model, test_sentence=question,
                                compare_sentences=compare_sentences)
    end = time.time()
    return end - start


if __name__ == '__main__':
    a = 1
