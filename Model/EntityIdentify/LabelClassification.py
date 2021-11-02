"""
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForSequenceClassification
import pandas as pd

from Model.Common import to_sts_sentence, get_similarity
from Static.Define import PathCommon
import time

# model = T5ForConditionalGeneration.from_pretrained('../Save/Try/A/')
model = BertForSequenceClassification.from_pretrained("../Save/T5classification/")
model.to('cpu')
# tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
# tokenConfig(tokenizer=tokenizer)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/Test/test1.csv"

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual"]
result_df = pd.DataFrame(columns=columns)
tqdm.pandas()


def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return [1.0, 2.0][probs.argmax()]


for index, row in tqdm(test_df.iterrows(), leave=False, total=len(result_df)):
    test_sentence = row["sentence"]
    sentence = row["sentence"]
    # sentence = "text classification " + test_sentence
    # sentence_ids = tokenizer(sentence, return_tensors='pt', padding=True)
    # out_ids = model.generate(input_ids=sentence_ids['input_ids'], attention_mask=sentence_ids['attention_mask'],
    #                          do_sample=False)
    # intent_group = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    intent_group = get_prediction(sentence)
    print(intent_group)
    try:
        actual = float(intent_group)
    except:
        print('not found')
        actual = -1
    max1 = actual
    max2 = actual
    max3 = actual
    new_row = {'test_id': row["test_index"], 'expected': row["intent_group_index"], 'max1': max1, 'max2': max2,
               'max3': max3}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf='../Result/A2/test_identity_result.csv', mode='w', index=False)
"""

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
            "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/Mining"
            "/answer_list.csv",
            header=0)
        test_sentence = row["sentence"]
        for i, r in temp_df.iterrows():
            compare_sentences = r["first"]
            similarity = get_similarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                        compare_sentences=compare_sentences)
            temp_df.loc[i, "similarity"] = similarity
        temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
        mean_df = temp_df.groupby(["label_index"])["similarity"].mean().reset_index().sort_values("similarity")
        max1 = mean_df.iloc[-1]
        max2 = mean_df.iloc[-2]
        max3 = mean_df.iloc[-3]
        new_row = {'test_id': row["sentence_index"], 'expected': row["intent_group_index"], 'actual': max1["label_index"],
                   'max2': max2["label_index"], 'max3': max3["label_index"]}
        result_df = result_df.append(new_row, ignore_index=True)
    result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
