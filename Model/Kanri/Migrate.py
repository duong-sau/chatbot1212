import pandas as pd
from tqdm import tqdm

from Static.Define import path_common

# define
n_group = 3

if __name__ == '__main__':
    sentence_df = pd.read_csv(path_common.sentence_list.value, header=0)
    sentence_df = sentence_df[(sentence_df.intent_group_index < n_group)]
    sentence_df.to_csv(path_common.sentence.value, index=False)

    intent_df = pd.read_csv(path_common.intent_list.value, header=0)
    intent_df = intent_df[(intent_df.intent_group_index < n_group)]
    intent_df.to_csv(path_common.intent.value, index=False)

    intent_group_df = pd.read_csv(path_common.intent_group_list.value, header=0)
    intent_group_df = intent_group_df[(intent_group_df.intent_group_index < n_group)]
    intent_group_df.to_csv(path_common.intent_group.value, index=False)

    exit()
