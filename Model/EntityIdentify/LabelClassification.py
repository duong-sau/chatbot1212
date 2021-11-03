
import pandas as pd
from tqdm.auto import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from os import path, mkdir

from Model.Common import get_similarity
from Static.Config import MODEL, tokenizer_config

names = ['CommandRefrence']
tqdm.pandas()
for name in names:
    model_path = '../CheckPoint/' + name + "/"
    result_path = './result.csv'
    if path.exists(result_path):
        raise "result exist -> duplicate run time"
    if not path.exists(model_path):
        raise "model not exist"

    tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
    tokenizer_config(tokenizer=tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.cpu()

    test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test" \
                ".csv "

    test_df = pd.read_csv(test_link, header=0)
    columns = ["test_id", "expected", "actual", "max2", "max3"]
    result_df = pd.DataFrame(columns=columns)

    for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
        temp_df = pd.read_csv(
            "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification"
            "/answer_list.csv",
            header=0)
        test_sentence = row["sentence"]
        for i, r in temp_df.iterrows():
            compare_sentences = r["first"]
            similarity1 = get_similarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                         compare_sentences=compare_sentences)
            try:
                compare_2 = r['second']
                similarity2 = get_similarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                             compare_sentences=compare_2)
                temp_df.loc[i, "similarity"] = (float(similarity1) + float(similarity2)) / 2
            except:
                temp_df.loc[i, "similarity"] = float(similarity1)

        temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
        mean_df = temp_df.groupby(["label_index"])["similarity"].mean().reset_index().sort_values("similarity")
        max1 = mean_df.iloc[-1]
        # max2 = mean_df.iloc[-2]
        # max3 = mean_df.iloc[-3]
        # new_row = {'test_id': row["sentence_index"], 'expected': row["intent_group_index"], 'actual': max1["label_index"],
        #           'max2': max2["label_index"], 'max3': max3["label_index"]}
        new_row = {'test_id': row["sentence_index"], 'expected': row["intent_group_index"],
                   'actual': max1["label_index"]}
        result_df = result_df.append(new_row, ignore_index=True)
    result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
