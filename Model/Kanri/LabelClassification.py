import pandas as pd
import numpy as np
from Static.Define import PathCommon

sentence_df = pd.read_csv('../Data/IntentClassification/LabelClassification/train.csv')

sentence_df['intent_group_index'] = sentence_df['intent_group_index']
sentence_df = sentence_df.rename(columns={'sentence': 'source', 'intent_group_index': 'target'})
sentence_df.to_csv(PathCommon.learn_data_label, index=False)
