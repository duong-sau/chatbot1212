import pandas as pd
from Static.Define import path_common
from tqdm import tqdm

if __name__ == '__main__':
    em_df = pd.DataFrame()
    data_df = pd.read_csv(path_common.data.value + "\\IntentIdentity\\sentence_list.csv")
    for index, row in tqdm(data_df.iterrows(), leave=False):
        sentence_1  = row["value"]
        for i, s in data_df.iterrows():
            sentence_2 = s["value"]
            out = 0
            if s["title"] == row["title"]:
                out = 1
            else:
                out = 0
            new = {"index": index, "sentence_1": sentence_1, "sentence_2": sentence_2, "out": out}
            em_df = em_df.append(new, ignore_index=True)
    em_df.to_csv("em.csv" )