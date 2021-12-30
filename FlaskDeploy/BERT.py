from transformers import BertForSequenceClassification, BertTokenizer
import time
import torch
import pandas as pd
from tqdm import tqdm

target_number = 28
target_name = []
for i in range(13, 39):
    target_name.append(i)

max_length = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=target_number)
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]
bert.to('cpu')


def get_prediction(sentence, top_k):
    # prepare our text into tokenized sequence
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to('cpu')
    outputs = bert(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    index = torch.topk(probs, top_k)[1][0].tolist()
    result = []
    for i in index:
        result.append(target_name[i])
    return result


# answer the question
def get_index_bert(question, top_k, s):
    s.sendall(bytes('clr-t5', "utf8"))
    s.sendall(bytes('clr-ln', "utf8"))
    result_df = pd.read_csv(
        "D:\\chatbot1212\\Model\\Data\\STSB\\sentence_list.csv",
        header=0)
    max_list = get_prediction(question, top_k)
    max_sentence_list = max_list
    return max_list, max_sentence_list
