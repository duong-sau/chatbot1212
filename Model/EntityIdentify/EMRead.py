from bert_embedding import BertEmbedding
from tqdm import tqdm
import pandas as pd
from Static.Define import path_common
import numpy as np
bert_embedding = BertEmbedding()


def getEM(sentence: str):
    sentences = sentence.split('\n')
    result = bert_embedding(sentences)
    return result[0][1]
# read data to compare
validate_df = pd.read_csv("tensor.csv")
# read intent list to merge
test_df = pd.read_table(filepath_or_buffer=path_common.data.value + "\\IntentIdentity\\EasyTest.tsv", header=0)
intent_df = pd.read_csv(path_common.data.value + "\\IntentIdentity\\intent_list.csv", header=0)
# create result data frame
result_df = pd.DataFrame()
columns = ["test_id", "value", "intent_id", "actual"]
tqdm.pandas()
validate_df["tensor"] = validate_df["tensor"].apply(eval).apply(np.array)
for index, row in tqdm(test_df.iterrows(), leave=False):
    test_sentence = row["question"]
    test_tensor = np.array(getEM(test_sentence)).astype(float)
    for i, s in validate_df.iterrows():
        validate_tensor = s['tensor'][1]
        #validate_tensor = s["tensor"]
        similarity = np.dot(test_tensor, validate_tensor)
        validate_df.loc[i, "similarity"] = similarity
    validate_df['similarity'] = pd.to_numeric(validate_df['similarity'], errors='coerce')
    mean_df = validate_df.groupby(["title"])["similarity"].mean().reset_index()
    merge_df = pd.merge(intent_df, mean_df, left_on="intent", right_on="title")
    max_row = merge_df.iloc[merge_df["similarity"].idxmax()]
    new_row = {'test_id': row["test_id"], 'value': test_sentence, 'intent_id': max_row["id"]}
    result_df = result_df.append(new_row, ignore_index=True)
    break
result_df.to_csv(path_or_buf='T5Identity.csv', mode='a')
