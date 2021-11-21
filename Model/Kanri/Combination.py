import socket
import time

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from FlaskDeploy.BM25 import word_token
from Model.Common import to_sts_sentence
from Static.Config import train_validate_test_split
from Static.Define import PathCommon

sentence_df = pd.read_csv(PathCommon.sentence)

train, test = train_validate_test_split(sentence_df)
test.to_csv(PathCommon.test, index=False)
train.to_csv(PathCommon.train, index=False)

train = pd.read_csv(PathCommon.train, header=0)

learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["label_index"] == second_row["label_index"]:
            stsb = stsb + 5.0
        else:
            pass
        source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=first_row["label"])
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=first_row["cluster"])
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data, index=False)
"""
learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["label_index"] == second_row["label_index"]:
            stsb = stsb + 5.0
        source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=first_row["label"])
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data_neg , index=False)
"""
"""
learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["cluster_index"] == second_row["cluster_index"]:
            stsb = stsb + 0.5
        if first_row["label_index"] == second_row["label_index"]:
            stsb = stsb + 4.5
        if first_row["label_index"] == second_row["label_index"] + 1 and first_row["cluster_index"] == \
                second_row["cluster_index"]:
            stsb = stsb + 0.5
        source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=("header: " + first_row["label"]))
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=("title: " + first_row["cluster"]))
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data_hed , index=False)
"""
# HOST = '127.0.0.1'
# PORT = 8000
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = (HOST, PORT)
# print('connecting to %s port ' + str(server_address))
# # s.connect(server_address)
# import random
#
# random.seed(1211)
# isDrop_list = [0, 1]
# drop_probability = (90, 20)
#
# sentences = pd.read_csv(PathCommon.train, header=0)
# input_corpus = sentences['sentence'].tolist()
#
# learn_data_df = pd.DataFrame()
# for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
#     # s.sendall(bytes('clr-t5', "utf8"))
#     docs = [first_row['sentence']] + input_corpus
#     docs = [word_token(d, lemma=True) for d in docs]
#     tokenized_corpus = [doc.split(' ') for doc in docs]
#     bm25 = BM25Okapi(tokenized_corpus[1:])
#     input_query = tokenized_corpus[0]
#     result_list = bm25.get_scores(input_query).tolist()
#
#     for second_index, second_row in train.iterrows():
#         score = result_list[second_index]
#         isDrop = bool(random.choices(isDrop_list, weights=drop_probability)[0])
#         stsb = 0
#         if first_row["cluster_index"] == second_row["cluster_index"]:
#             stsb = stsb + 0.4
#         if first_row["label_index"] == second_row["label_index"]:
#             stsb = stsb + 4.6
#
#         if first_row["label_index"] == second_row["label_index"] + 1 and first_row["cluster_index"] == \
#                 second_row["cluster_index"]:
#             stsb = stsb + 2.2
#         if first_row["label_index"] == second_row["label_index"] - 1 and first_row["cluster_index"] == \
#                 second_row["cluster_index"]:
#             stsb = stsb + 2.2
#
#         if first_row["label_index"] == second_row["label_index"] + 2 and first_row["cluster_index"] == \
#                 second_row["cluster_index"]:
#             stsb = stsb + 1
#
#         if first_row["label_index"] == second_row["label_index"] - 2 and first_row["cluster_index"] == \
#                 second_row["cluster_index"]:
#             stsb = stsb + 1
#
#         if (stsb == 0 or stsb == 0.4) and isDrop:
#             continue
#         if first_row["cluster_index"] == second_row["cluster_index"]:
#             stsb += min(score, 10) / 1.75
#         else:
#             stsb += min(score, 10) / 3.5
#         if stsb >= 5.0:
#             stsb = 5.0
#         stsb = str(round(stsb, 1))
#         # time.sleep(0.075)
#         # s.sendall(bytes('t5_' + stsb, "utf8"))
#         source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
#         new = {"source": source, 'target': str(stsb)}
#         learn_data_df = learn_data_df.append(new, ignore_index=True)
#     source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=("header: " + first_row["label"]))
#     new = {"source": source, 'target': '5.0'}
#     learn_data_df = learn_data_df.append(new, ignore_index=True)
#     source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=("title: " + first_row["cluster"]))
#     new = {"source": source, 'target': '3.8'}
#     learn_data_df = learn_data_df.append(new, ignore_index=True)
# learn_data_df = learn_data_df.round(1)
# learn_data_df.to_csv(PathCommon.learn_data, index=False)
