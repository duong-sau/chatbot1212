import pandas as pd
from tqdm import tqdm

from Model.Common import toT5sentence, train_validate_test_split
from Static.Define import path_common


sentence_df = pd.read_csv(path_common.sentence.value)

sentence_df['sentence'] = 'text classification: ' + sentence_df['sentence']
sentence_df['intent_group_index'] = sentence_df['intent_group_index'] + 10
sentence_df = sentence_df.rename(columns={'sentence': 'source', 'intent_group_index': 'target'})
sentence_df.to_csv(path_common.learn_data_a.value, index=False)
