import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Static.Define import path_common


def train_validate_test_split(df, train_percent=.8, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.iloc[perm[:train_end]]
    test = df.iloc[perm[train_end:]]
    return train, test


sentence_df = pd.read_csv(path_common.sentence.value)
intent_df = pd.read_csv(path_common.intent.value, header=0)
intent_group_df = pd.read_csv(path_common.intent_group.value, header=0)
train, test = train_validate_test_split(sentence_df)
learn_data_df = pd.DataFrame()
data_index = 0
prefix = 'stsb '
sentence_1 = 'sentence1: '
sentence_2 = '. sentence2: '
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["intent_group_index"] == second_row["intent_group_index"]:
            stsb = stsb + 0.5
        if first_row["intent_index"] == second_row["intent_index"]:
            stsb = stsb + 4.5
        if first_row["intent_index"] == second_row["intent_index"] + 1 and first_row["intent_group_index"] == \
                second_row["intent_group_index"]:
            stsb = stsb + 0.5
        source = prefix + sentence_1 + first_row["sentence"] + sentence_2 + second_row["sentence"]
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = prefix + sentence_1 + first_row["sentence"] + sentence_2 + first_row["intent"]
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = prefix + sentence_1 + first_row["sentence"] + sentence_2 + first_row["intent_group"]
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(path_common.learn_data_pos.value, index=False)
train.to_csv(path_common.train_pos.value, index=False)
test.to_csv(path_common.test_pos.value, index=False)
exit()
