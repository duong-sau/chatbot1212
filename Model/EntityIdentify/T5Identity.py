from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

from Model.Common import toT5sentence, getSimilarity
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import path_common
import time

model = T5ForConditionalGeneration.from_pretrained(path_common.model.value + '\\' + MODEL['name'] + "\\2Layer")
preModel = T5ForConditionalGeneration.from_pretrained(path_common.model.value + '\\' + MODEL['name'] + "\\AClass")
model.to('cpu')
tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
tokenConfig(tokenizer=tokenizer)

test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test.csv"

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual"]
result_df = pd.DataFrame(columns=columns)
tqdm.pandas()
for index, row in tqdm(test_df.iterrows(), leave=False, total=len(result_df)):
    temp_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
        header=0)
    test_sentence = row["sentence"]
    sentence = "text classification: " + test_sentence
    sentence_ids = tokenizer(sentence, return_tensors='pt', padding=True)
    out_ids = preModel.generate(input_ids=sentence_ids['input_ids'], attention_mask=sentence_ids['attention_mask'],
                             do_sample=False)
    intent_group = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    actual = float(intent_group) - 10
    temp_df = temp_df[temp_df['intent_group_index'] == actual]
    for i, r in temp_df.iterrows():
        compare_sentences = r["sentence"]
        similarity = getSimilarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                   compare_sentences=compare_sentences)
        temp_df.loc[i, "similarity"] = similarity
    temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
    path_name = "../Result/2-1/" + str(row['sentence_index']) + "_test.csv"
    temp_df.to_csv(path_name, index=0)
    mean_df = temp_df.groupby(["intent_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max1 = mean_df.iloc[-1]
    max2 = mean_df.iloc[-2]
    max3 = mean_df.iloc[-3]
    new_row = {'test_id': row["sentence_index"], 'expected': row["intent_index"], 'max1': max1["intent_index"],
               'max2': max2["intent_index"], 'max3': max3["intent_index"]}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf='../Result/2-1/test_identity_result.csv', mode='w', index=False)
