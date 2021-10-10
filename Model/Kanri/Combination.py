import pandas as pd
from tqdm import tqdm

from Model.Common import toT5sentence, train_validate_test_split
from Static.Define import path_common


sentence_df = pd.read_csv(path_common.sentence.value)
intent_df = pd.read_csv(path_common.intent.value, header=0)
intent_group_df = pd.read_csv(path_common.intent_group.value, header=0)
train, test = train_validate_test_split(sentence_df)
test.to_csv(path_common.test.value, index=False)
train.to_csv(path_common.train.value, index=False)

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
        source = toT5sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = toT5sentence(sentence1=first_row["sentence"], sentence2=first_row["intent"])
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = toT5sentence(sentence1=first_row["sentence"], sentence2=first_row["intent_group"])
    new = {"source": source, 'target': '3.8'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(path_common.learn_data_pos.value, index=False)

learn_data_df = pd.DataFrame()
for first_index, first_row in tqdm(train.iterrows(), total=len(train.index)):
    for second_index, second_row in train.iterrows():
        stsb = 0
        if first_row["intent_index"] == second_row["intent_index"]:
            stsb = stsb + 5.0
        source = toT5sentence(sentence1=first_row["sentence"], sentence2=second_row["sentence"])
        new = {"source": source, 'target': str(stsb)}
        learn_data_df = learn_data_df.append(new, ignore_index=True)
    source = toT5sentence(sentence1=first_row["sentence"], sentence2=first_row["intent"])
    new = {"source": source, 'target': '5.0'}
    learn_data_df = learn_data_df.append(new, ignore_index=True)
learn_data_df.to_csv(path_common.learn_data_neg.value, index=False)
exit()


