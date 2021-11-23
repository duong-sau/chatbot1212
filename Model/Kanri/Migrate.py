import pandas as pd
from tqdm import tqdm

from Static.Define import PathCommon

# define
n_group = [3.0, 4.0]

if __name__ == '__main__':
    sentence_df = pd.read_csv(PathCommon.sentence_list, header=0)
    sentence_df = sentence_df[(sentence_df.cluster_index.isin(n_group))]
    sentence_df.to_csv(PathCommon.sentence, index=False)

    label_df = pd.read_csv(PathCommon.label_list, header=0)
    label_df = label_df[(label_df.cluster_index.isin(n_group))]
    label_df.to_csv(PathCommon.label, index=False)

    cluster_df = pd.read_csv(PathCommon.cluster_list, header=0)
    cluster_df = cluster_df[(cluster_df.cluster_index.isin(n_group))]
    cluster_df.to_csv(PathCommon.cluster, index=False)

    answer_df = pd.read_csv(PathCommon.answer_list, header=0)
    answer_df = answer_df[(answer_df.cluster_index.isin(n_group))]
    answer_df = answer_df.to_csv(PathCommon.answer, index=False)
    print(sentence_df.label_index.value_counts())
    exit()
