from bert_embedding import BertEmbedding
from tqdm import tqdm
import pandas as pd

from Static.Define import path_common

bert_embedding = BertEmbedding()


def GetEmbedding(sentence: str):
    sentences = sentence.split('\n')
    result = bert_embedding(sentences)
    return result[0][1]


data_df = pd.read_csv(path_common.data.value + "\\IntentIdentity\\sentence_list.csv")
result_df = pd.DataFrame()
# read intent list to merge
tqdm.pandas()
# for index, row in tqdm(test_df.iterrows(), leave=False):
for index, row in tqdm(data_df.iterrows()):
    sentences = row["value"]
    tensor = GetEmbedding(sentences)
    new = {"title": row["title"], "value": row["value"], "tensor": tensor}
    result_df = result_df.append(new, ignore_index=True)
result_df.to_csv("tensor.csv")
