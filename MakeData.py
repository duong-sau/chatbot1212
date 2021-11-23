import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy

from Static.Config import MODEL

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
data = pd.read_csv("D:\\chatbot1212\\Model\\Data\\IntentClassification\\sentence_list.csv", header=0)
result = numpy.array()
for index, row in tqdm(data.iterrows(), leave=False, total=len(data)):
    m = "helo how are you"
    ids = tokenizer(m, return_tensors="pt", padding=True)
    output = model.encoder(input_ids=ids['input_ids'], attention_mask=ids['attention_mask'], return_dict=True)
    pooled_sentence = torch.mean(output.last_hidden_state, dim=1).squeeze().detach().numpy()
    numpy.concatenate(result, pooled_sentence, axis=0)
numpy.savetxt('markers.txt', result)
