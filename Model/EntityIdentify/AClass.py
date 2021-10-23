from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

from Model.Common import toT5sentence, getSimilarity
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import path_common
import time

model = T5ForConditionalGeneration.from_pretrained(path_common.model.value + '\\' + MODEL['name'] + "\\A2")
model.to('cpu')
tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
tokenConfig(tokenizer=tokenizer)

test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test.csv"

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual"]
result_df = pd.DataFrame(columns=columns)
tqdm.pandas()
for index, row in tqdm(test_df.iterrows(), leave=False, total=len(result_df)):
    test_sentence = row["sentence"]
    sentence = "text classification: " + test_sentence
    sentence_ids = tokenizer(sentence, return_tensors='pt', padding=True)
    out_ids = model.generate(input_ids=sentence_ids['input_ids'], attention_mask=sentence_ids['attention_mask'],
                             do_sample=False)
    intent_group = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    try:
        actual = float(intent_group)
    except:
        print('not found')
        actual = -1
    max1 = actual
    max2 = actual
    max3 = actual
    new_row = {'test_id': row["sentence_index"], 'expected': row["intent_group_index"], 'max1': max1, 'max2': max2,
               'max3': max3}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf='../Result/A2/test_identity_result.csv', mode='w', index=False)
