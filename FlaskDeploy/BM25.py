import time

import pandas as pd
from rank_bm25 import BM25Okapi
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = list(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def word_token(tokens, lemma=False):
    tokens = str(tokens)
    tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
    tokens = re.sub(r"\s+", " ", tokens)
    if lemma:
        return " ".join([lemmatizer.lemmatize(token, 'v') for token in word_tokenize(tokens.lower()) if
                         token not in stop_words and token.isalpha()])
    else:
        return " ".join(
            [token for token in word_tokenize(tokens.lower()) if token not in stop_words and token.isalpha()])


def get_index_bm25(input_query, top_k, s):
    s.sendall(bytes('clr-bm', "utf8"))
    result_df = pd.read_csv(
        "D:\\chatbot1212\\Model\\Data\\IntentClassification\\sentence_list.csv",
        header=0)
    input_corpus = result_df['sentence'].tolist()
    docs = [input_query] + input_corpus
    docs = [word_token(d, lemma=True) for d in docs]
    tokenized_corpus = [doc.split(' ') for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus[1:])
    input_query = tokenized_corpus[0]
    result_list = bm25.get_scores(input_query).tolist()
    for similarity in result_list:
        time.sleep(0.075)
        s.sendall(bytes('bm_' + str(format(similarity / 5, '.1f')), "utf8"))
    result_df['similarity'] = result_list
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
