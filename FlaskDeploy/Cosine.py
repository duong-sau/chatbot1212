import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


def embed(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence.lower(), add_special_tokens=True)[:512]).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)[0]
        res = torch.mean(outputs, dim=1).detach().cpu().numpy()
    return res[0]


def get_scores(query, corpus, group):
    emb_corpus = embed(corpus)
    emb_query = embed(query)
    result_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list"
        ".csv",
        header=0)
    result_df = result_df[result_df['intent_group_index'].isin(group)]
    for i, r in tqdm(result_df.iterrows(), total=len(result_df)):
        compare_sentences = r["sentence"]
        similarity = cosine_similarity(emb_query, emb_corpus)[0]
        result_df.loc[i, "similarity"] = similarity
    result_df['similarity'] = pd.to_numeric(result_df['similarity'], errors='coerce')
    mean_df = result_df.groupby(["intent_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max_list = []
    max_sentence_list = []
    try:
        max_list = mean_df.iloc[-5:]['intent_index'].tolist()
        max_list.reverse()
        for max_id in max_list:
            group_df = result_df[result_df['intent_index'] == max_id]
            idx = group_df['similarity'].idxmax()
            max_sentence_list.append(result_df.iloc[idx]['sentence'][:300])
    except ValueError:
        index = -1
    return max_list, max_sentence_list



if __name__ == '__main__':
    sentence_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification"
        "/sentence_list.csv",
        header=0)
    input_query = 'How can I perform bootstrap analysis'
    input_corpus = sentence_df['sentence'].tolist()
    out = get_scores(query=input_query, corpus=input_corpus)
    print(out)
