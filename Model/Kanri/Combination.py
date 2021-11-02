import pandas as pd
from tqdm import tqdm

from Model.Common import to_sts_sentence
from Static.Config import train_validate_test_split
from Static.Define import PathCommon

sentence_df = pd.read_csv(PathCommon.sentence)
# intent_df = pd.read_csv(PathCommon.intent, header=0)
# intent_group_df = pd.read_csv(PathCommon.intent_group, header=0)

train, test = train_validate_test_split(sentence_df)
test.to_csv(PathCommon.test, index=False)
train.to_csv(PathCommon.train, index=False)

train = pd.read_csv(PathCommon.train, header=0)

learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["intent_index"] == second_row["intent_index"]:
            stsb = stsb + 5.0
        source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=first_row["intent"])
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = to_sts_sentence(sentence1=first_row["sentence"], sentence2=first_row["intent_group"])
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data_pos, index=False)
"""
learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["intent_index"] == second_row["intent_index"]:
            stsb = stsb + 5.0
        source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=first_row["intent"])
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data_neg , index=False)
"""
"""
learn_data_df = pd.DataFrame()
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
        source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=("header: " + first_row["intent"]))
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=("title: " + first_row["intent_group"]))
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data_hed , index=False)
"""
"""
import random
random.seed(1211)
isAdd_list = [0, 1]
add_probability = (30, 70)
isDrop_list = [0, 1]
drop_probability = (40, 60)

learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    add_df = train[train['intent_index'] == first_row['intent_index']]
    for i in range(len(add_df)):
        isAdd = (random.choices(isAdd_list, weights=add_probability)[0])
        if isAdd:
            source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=add_df.iloc[i]["sentence"] + " moreover "+ first_row["sentence"])
            new = {"source": source, 'target': str(4.8)}
            learn_data_df = learn_data_df.append(new, ignore_index=True)
        pass
    for second_index, second_row in train.iterrows():
        isDrop = bool(random.choices(isDrop_list, weights=drop_probability)[0])
        stsb = 0
        if first_row["intent_group_index"] == second_row["intent_group_index"]:
            stsb = stsb + 0.4
        if first_row["intent_index"] == second_row["intent_index"]:
            stsb = stsb + 4.6
        
        if first_row["intent_index"] == second_row["intent_index"] + 1 and first_row["intent_group_index"] == \
                second_row["intent_group_index"]:
            stsb = stsb + 0.5
        
        if (stsb == 0  or stsb == 0.4)and isDrop:
            continue
        source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=("header: " + first_row["intent"]))
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source =  to_sts_sentence(sentence1=first_row["sentence"], sentence2=("title: " + first_row["intent_group"]))
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(PathCommon.learn_data_han , index=False)
"""
