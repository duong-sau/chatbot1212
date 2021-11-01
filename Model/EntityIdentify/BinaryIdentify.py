from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from Model.EntityIdentify.MeanSum import MeanSum
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import PathCommon

model = T5ForConditionalGeneration.from_pretrained("../Save/Try/freeze3/")
model.to('cpu')
tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
tokenConfig(tokenizer=tokenizer)

test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test.csv"
temp_df = pd.read_csv(
    "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
    header=0)
temp_df['similarity'] = float('NaN')
test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual"]
result_df = pd.DataFrame(columns=columns)
for i, r in tqdm(test_df.iterrows()):
    test_sentence = r['sentence']
    sub_df = MeanSum(test_sentence=test_sentence, tokenizer=tokenizer, model=model, temp_df=temp_df, depth=2,
                     return_max=False)
    sub_df = MeanSum(test_sentence=test_sentence, tokenizer=tokenizer, model=model, temp_df=sub_df, depth=5,
                     return_max=False)
    intent_id = MeanSum(test_sentence=test_sentence, tokenizer=tokenizer, model=model, temp_df=sub_df, depth=2,
                        return_max=True)
    new_row = {'test_id': r["sentence_index"], 'expected': intent_id, 'actual': r["intent_index"]}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf='../Result/1-layer-test_identity_result.csv', mode='w', index=False)
