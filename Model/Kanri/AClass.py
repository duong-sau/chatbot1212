import pandas as pd
import numpy as np
from Static.Define import PathCommon

sentence_df = pd.read_csv(PathCommon.train.value)

sentence_df['intent_group_index'] = sentence_df['intent_group_index']
sentence_df = sentence_df.rename(columns={'sentence': 'source', 'intent_group_index': 'target'})
sentence_df.to_csv(PathCommon.learn_data_a.value, index=False)
