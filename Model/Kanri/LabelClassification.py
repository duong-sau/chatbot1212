from tqdm import tqdm

from Static.Define import PathCommon
import pandas as pd
from rank_bm25 import BM25Okapi
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

sentence_df = pd.read_csv('D:\\chatbot1212\\Model\\Cluster\\Mining\\sentence_list.csv')
from Static.Config import train_validate_test_split
from Static.Define import PathCommon

train, test = train_validate_test_split(sentence_df)
test.to_csv(PathCommon.test, index=False)
train.to_csv(PathCommon.train, index=False)

# test = pd.read_csv(PathCommon.test, header=0)
# train = pd.read_csv(PathCommon.train, header=0)

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


def get_index_bm25(input_query, top_k):
    result_df = pd.read_csv(
        "D:\\chatbot1212\\Model\\Cluster\\Mining\\sentence_list.csv",
        header=0)
    input_corpus = result_df['sentence'].tolist()
    docs = [input_query] + input_corpus
    docs = [word_token(d, lemma=True) for d in docs]
    tokenized_corpus = [doc.split(' ') for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus[1:])
    input_query = tokenized_corpus[0]
    result_list = bm25.get_scores(input_query).tolist()

    result_df['similarity'] = result_list
    mean_df = result_df.groupby(["cluster_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max_list = []
    try:
        max_list = mean_df.iloc[-top_k:]['cluster_index'].tolist()
        max_list.reverse()
    except ValueError:
        max_list = []
    return max_list


# for i, r in tqdm(train.iterrows(), total=1035):
#     max_l = get_index_bm25(r['sentence'], 2)
#     t = str(r.cluster_index) + " " + str(max_l[0]) + " " + str(max_l[1])
#     train.at[i, 'target'] = t

train.to_csv(PathCommon.learn_data, index=False)
