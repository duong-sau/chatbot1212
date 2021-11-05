import pandas as pd
from tqdm.auto import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from os import path, mkdir

from Model.Common import get_similarity
from Static.Config import MODEL, tokenizer_config

names = []
for i in range(64):
    if i == 24:
        continue
    bString = bin(i)[2:].zfill(6)
    names.append(bString)

tqdm.pandas()
for name in names:
    model_path = '../CheckPoint/' + name + "/"
    result_dir = '../Result/' + name
    result_path = result_dir + '/result.csv'
    if not path.exists(result_dir):
        mkdir(result_dir)
    if path.exists(result_path):
        print("result exist -> duplicate run time: ", str(int(name, 2)))
        continue
    else:
        print('start run on runtime:               ', str(int(name, 2)))
    if not path.exists(model_path):
        print('Model not found in runtime:         ', str(int(name, 2)))
        continue

    tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
    tokenizer_config(tokenizer=tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.cpu()

    test_link = "C:\\Users\\Sau\\IdeaProjects\\chatbot1212\\Model\\Data\\IntentClassification\\Tutorial\\test.csv"

    test_df = pd.read_csv(test_link, header=0)
    columns = ["test_id", "expected", "actual", "max2", "max3"]
    result_df = pd.DataFrame(columns=columns)

    for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
        temp_df = pd.read_csv(
            "C:\\Users\\Sau\\IdeaProjects\\chatbot1212\\Model\\Data\\IntentClassification\\Tutorial\\sentence_list.csv",
            header=0)
        test_sentence = row["sentence"]
        for i, r in temp_df.iterrows():
            compare_sentences = r["sentence"]
            similarity = get_similarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                        compare_sentences=compare_sentences)
            temp_df.loc[i, "similarity"] = similarity
        temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
        mean_df = temp_df.groupby(["intent_index"])["similarity"].mean().reset_index().sort_values("similarity")
        max1 = mean_df.iloc[-1]
        max2 = mean_df.iloc[-2]
        max3 = mean_df.iloc[-3]
        new_row = {'test_id': row["sentence_index"], 'expected': row["intent_index"], 'actual': max1["intent_index"],
                   'max2': max2["intent_index"], 'max3': max3["intent_index"]}
        result_df = result_df.append(new_row, ignore_index=True)
    result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
