import pandas as pd
from tqdm import tqdm

from Static.Define import PathCommon

# define
n_group = 3

if __name__ == '__main__':
    sentence_df = pd.read_csv(PathCommon.sentence_list.value, header=0)
    sentence_df = sentence_df[(sentence_df.intent_group_index < n_group)]
    sentence_df.to_csv(PathCommon.sentence.value, index=False)

    intent_df = pd.read_csv(PathCommon.intent_list.value, header=0)
    intent_df = intent_df[(intent_df.intent_group_index < n_group)]
    intent_df.to_csv(PathCommon.intent.value, index=False)

    intent_group_df = pd.read_csv(PathCommon.intent_group_list.value, header=0)
    intent_group_df = intent_group_df[(intent_group_df.intent_group_index < n_group)]
    intent_group_df.to_csv(PathCommon.intent_group.value, index=False)

    exit()
