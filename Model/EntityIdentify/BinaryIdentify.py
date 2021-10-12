from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

from Model.Common import toT5sentence, getSimilarity
from Model.EntityIdentify.MeanSum import MeanSum
from Model.FineTurn.Define import MODEL, tokenConfig
from Static.Define import path_common
import time
model = T5ForConditionalGeneration.from_pretrained(path_common.model.value + '\\' + MODEL['name'] + "\\POS")
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
start_time = time.time()
temp_df = MeanSum(test_sentence="To reduce computational burden, one can use the option -mset to restrict the testing procedure to a subset of base models instead of testing the entire set of all available models. For example, -mset WAG,LG will test only models like WAG+... or LG+.... Another useful option in this respect is -msub for AA dat", tokenizer=tokenizer, model=model,temp_df=temp_df, depth=2, return_max=False)
temp_df = MeanSum(test_sentence="To reduce computational burden, one can use the option -mset to restrict the testing procedure to a subset of base models instead of testing the entire set of all available models. For example, -mset WAG,LG will test only models like WAG+... or LG+.... Another useful option in this respect is -msub for AA dat", tokenizer=tokenizer, model=model,temp_df=temp_df, depth=4, return_max=False)
i = MeanSum(test_sentence="To reduce computational burden, one can use the option -mset to restrict the testing procedure to a subset of base models instead of testing the entire set of all available models. For example, -mset WAG,LG will test only models like WAG+... or LG+.... Another useful option in this respect is -msub for AA dat", tokenizer=tokenizer, model=model,temp_df=temp_df, depth=2, return_max=True)
end_time = time.time()
print(start_time - end_time)

