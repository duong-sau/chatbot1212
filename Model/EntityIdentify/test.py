import pandas as pd
import torch
from transformers import BertForSequenceClassification
from Model.Common import toT5sentence
from Static.Define import path_common

"""
train = pd.read_csv(path_common.train.value, header = 0)
test = pd.read_csv(path_common.test.value, header = 0)
list = train['sentence_index'].tolist()
result = test[test['sentence_index'].isin(list)]
"""
import torch.nn as nn
from transformers import AutoModel


class PosModel(BertForSequenceClassification):
    def __init__(self, config):
        super(PosModel, self).__init__(config)

        self.base_model = AutoModel.from_pretrained('t5-small')
        self.dropout = nn.Dropout(0.05)
        self.linear = nn.Linear(512, 2)  # output features from bert is 768 and 2 is ur number of labels

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, decoder_input_ids=torch.tensor([0]).unsqueeze(0))
        # You write you new head here
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)
        return outputs


model = PosModel()
model.cpu()
model.eval()
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
test_sentence = 'hello'
compare_sentences = 'hi'
T5_format_sentence = toT5sentence(sentence1=test_sentence, sentence2=compare_sentences)
enc = tokenizer(T5_format_sentence, return_tensors="pt")
logits = model(**enc)[0]
print(logits)