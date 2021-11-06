import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from FlaskDeploy.BM25 import word_token

tqdm.pandas()


def get_index_bm25(input_query, top_k):
    result_df = pd.read_csv("..\\Data\\IntentClassification\\sentence_list.csv", header=0)
    input_corpus = result_df['sentence'].tolist()
    docs = [input_query] + input_corpus
    docs = [word_token(d, lemma=True) for d in docs]
    tokenized_corpus = [doc.split(' ') for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus[1:])
    input_query = tokenized_corpus[0]
    result_df['similarity'] = bm25.get_scores(input_query)
    mean_df = result_df.groupby(["intent_group_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max_list = []
    max_sentence_list = []
    try:
        max_list = mean_df.iloc[-top_k:]['intent_group_index'].tolist()
        max_list.reverse()
        for max_id in max_list:
            group_df = result_df[result_df['intent_group_index'] == max_id]
            idx = group_df['similarity'].idxmax()
            max_sentence_list.append(result_df.iloc[idx]['sentence'][:300])
    except ValueError:
        index = -1
    return max_list, max_sentence_list


result_path = './result.csv'

test_link = "..\\Data\\IntentClassification\\test.csv "

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual", "max2", "max3"]
result_df = pd.DataFrame(columns=columns)

for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
    test_sentence = row['sentence']
    max_list = get_index_bm25(test_sentence, 2)[0]
    new_row = {'test_id': row["sentence_index"], 'expected': row["intent_group_index"], 'actual': max_list[0],
               'max2': max_list[1], 'max3':0}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
