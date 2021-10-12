import pandas as pd
from tqdm import tqdm

from Model.Common import getSimilarity


# temp_df = pd.read_csv(
#     "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
#     header=0)

def MeanSum(test_sentence, temp_df, depth, tokenizer, model, return_max=True):
    for i, r in temp_df.iterrows():
        if not pd.isnull(r['similarity']):
            continue
        if r['sub_index'] > depth and not return_max:
            break
        compare_sentences = r["sentence"]
        similarity = getSimilarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                   compare_sentences=compare_sentences)
        temp_df.loc[i, "similarity"] = similarity
    temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
    mean_df = temp_df.groupby(["intent_index"])["similarity"].mean().reset_index()
    max_row = mean_df.iloc[mean_df["similarity"].idxmax()]
    if return_max:
        return max_row["intent_index"]
    else:
        mean_df = mean_df.sort_values("similarity").reset_index()
        length = len(mean_df)
        series = mean_df[0:int(length/2)]['intent_index']
        remove_list = series.tolist()
        temp_df = temp_df[~(temp_df['intent_index'].isin(remove_list))]
        return temp_df
