# group classification
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from FlaskDeploy.BM25 import word_token
from Model.Common import get_similarity
from Static.Config import get_device, MODEL, tokenizer_config

device = get_device()

# import model
siamese_tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
tokenizer_config(tokenizer=siamese_tokenizer)
siamese_model = T5ForConditionalGeneration.from_pretrained('D:\chatbot1212\Model\CheckPoint\Command')
siamese_model.to(device)


def get_cluster(input_query, top_p):
    result_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
        header=0)
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
    except ValueError:
        print('exception')
    return max_list


# answer the question
def get_index(question, group, top_k, x_values, y_values):
    result_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list"
        ".csv",
        header=0)
    result_df = result_df[result_df['cluster_index'].isin(group)].reset_index()
    for i, r in tqdm(result_df.iterrows(), total=len(result_df)):
        compare_sentences = r["sentence"]
        similarity = get_similarity(tokenizer=siamese_tokenizer, model=siamese_model, test_sentence=question,
                                    compare_sentences=compare_sentences)
        x_values.append(i)
        y_values.append(similarity)
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

