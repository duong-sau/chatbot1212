from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

from Model.Common import toT5sentence
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import path_common

model = T5ForConditionalGeneration.from_pretrained(path_common.model.value + '\\' + MODEL['name'] + "\\T5STS-POS")
model.to('cpu')
tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])
tokenConfig(tokenizer=tokenizer)

test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test.csv"

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual"]
result_df = pd.DataFrame(columns=columns)
task_prefix = 'stsb '
tqdm.pandas()
for index, row in tqdm(test_df.iterrows(), leave=False, total=len(result_df)):
    temp_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
        header=0)
    test_sentence = row["sentence"]
    for i, r in temp_df.iterrows():
        compare_sentences = r["sentence"]
        T5_format_sentence = toT5sentence(sentence1=test_sentence, sentence2=compare_sentences)
        inputs = tokenizer(T5_format_sentence, return_tensors="pt", padding=True)
        output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                          do_sample=False)
        similarity = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        temp_df.loc[i, "similarity"] = similarity
    temp_df.to_csv(str(row["sentence_index"]) + "_test.csv", mode='w', index=False)
    temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
    mean_df = temp_df.groupby(["intent_index"])["similarity"].mean().reset_index()
    max_row = mean_df.iloc[mean_df["similarity"].idxmax()]
    new_row = {'test_id': row["sentence_index"], 'expected': max_row["intent_index"], 'actual': row["intent_index"]}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf='test_identity_result.csv', mode='w', index=False)
