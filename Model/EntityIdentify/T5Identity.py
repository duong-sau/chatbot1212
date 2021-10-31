from pytorch_lightning.utilities import warnings
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, T5Config
import pandas as pd
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput

from Model.Common import toT5sentence, getSimilarity
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import path_common
import time

# define
# names = ["freeze'1", "freeze'2", "freeze'3", "freeze'4", "freeze'5", "freeze'6"]
names = ['freeze3']

for name in names:
    tokenizer = T5Tokenizer.from_pretrained(MODEL['name'])

    tokenConfig(
        tokenizer=tokenizer)
    model = T5ForConditionalGeneration.from_pretrained('../Save/Try/' + name + "/")
    model.cpu()

    test_link = "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/test.csv"

    test_df = pd.read_csv(test_link, header=0)
    columns = ["test_id", "expected", "actual", "max2", "max3"]
    result_df = pd.DataFrame(columns=columns)
    tqdm.pandas()
    for index, row in tqdm(test_df.iterrows(), leave=False, total=len(result_df)):
        temp_df = pd.read_csv(
            "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list.csv",
            header=0)
        test_sentence = row["sentence"]
        for i, r in temp_df.iterrows():
            compare_sentences = r["sentence"]
            similarity = getSimilarity(tokenizer=tokenizer, model=model, test_sentence=test_sentence,
                                       compare_sentences=compare_sentences)
            temp_df.loc[i, "similarity"] = similarity
        temp_df['similarity'] = pd.to_numeric(temp_df['similarity'], errors='coerce')
        path_name = "../Result/" + name + "/" + str(row['sentence_index']) + "_test.csv"
        temp_df.to_csv(path_name, index=0)
        mean_df = temp_df.groupby(["intent_index"])["similarity"].mean().reset_index().sort_values("similarity")
        max1 = mean_df.iloc[-1]
        max2 = mean_df.iloc[-2]
        max3 = mean_df.iloc[-3]
        new_row = {'test_id': row["sentence_index"], 'expected': row["intent_index"], 'actual': max1["intent_index"],
                   'max2': max2["intent_index"], 'max3': max3["intent_index"]}
        result_df = result_df.append(new_row, ignore_index=True)
    result_df.to_csv(path_or_buf='../Result/' + name + '/test_identity_result.csv', mode='w', index=False)
