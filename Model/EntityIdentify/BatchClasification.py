from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import pandas as pd

from Model.Common import toT5sentence, getSimilarity
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import path_common

model = T5ForConditionalGeneration.from_pretrained(path_common.model.value + '\\' + MODEL['name'] + "\\2Layer")
model.to('cpu')
tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
tokenConfig(tokenizer=tokenizer)

test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test.csv"

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual"]
result_df = pd.DataFrame(columns=columns)
tqdm.pandas()
sentence_df = pd.read_csv(
    "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
    header=0)
temp_df = pd.DataFrame()
groups = list()
for g, data in sentence_df.groupby('intent_index'):
    print(g, data)
    groups.append(data)

for index, row in tqdm(test_df.iterrows(), leave=False, total=len(result_df)):
    test_sentence = row["sentence"]
    for sentences in groups:
        compare_sentences = sentences["sentence"]
        input_ids = tokenizer.batch_encode_plus(compare_sentences, return_tensors='pt', padding=True)
        similaritys = model.generate(**input_ids)
        similaritys = tokenizer.batch_decode(similaritys, skip_special_tokens=True)
        new_ = {'intent_index': groups[0]['intent_index'], 'mean': sum(int(similaritys)) / len(similaritys)}
        temp_df = temp_df.append(new_)
    mean_df = temp_df.sort_values("mean")
    max1 = mean_df.iloc[-1]
    max2 = mean_df.iloc[-2]
    max3 = mean_df.iloc[-3]
    new_row = {'test_id': row["sentence_index"], 'expected': row["intent_index"], 'max1': max1["intent_index"],
               'max2': max2["intent_index"], 'max3': max3["intent_index"]}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf='../Result/Batch_/test_identity_result.csv', mode='w', index=False)
