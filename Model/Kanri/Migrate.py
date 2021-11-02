import pandas as pd
from tqdm import tqdm

from Static.Define import PathCommon

# define
n_group = [3.0, 4.0]

if __name__ == '__main__':
    sentence_df = pd.read_csv(PathCommon.sentence_list, header=0)
    sentence_df = sentence_df[(sentence_df.intent_group_index.isin(n_group))]
    sentence_df.to_csv(PathCommon.sentence, index=False)

    intent_df = pd.read_csv(PathCommon.intent_list, header=0)
    intent_df = intent_df[(intent_df.intent_group_index.isin(n_group))]
    intent_df.to_csv(PathCommon.intent, index=False)

    intent_group_df = pd.read_csv(PathCommon.intent_group_list, header=0)
    intent_group_df = intent_group_df[(intent_group_df.intent_group_index.isin(n_group))]
    intent_group_df.to_csv(PathCommon.intent_group, index=False)

    answer_df = pd.read_csv(PathCommon.answer_list, header=0)
    answer_df = answer_df[(answer_df.label_index.isin(n_group))]
    answer_df = answer_df.to_csv(PathCommon.answer, index=False)
    exit()
