import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

from Static.Define import path_common

list_pass = [73, 193, 91]
list_fail = [89, 23, 169]

if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(path_common.model.value + "\\Save\\T5STS-POS")
    sentence_df = pd.read_csv(path_common.sentence_list.value, header = 0)

    pass_sentence = sentence_df[sentence_df['sentence_index'] == list_pass[0]].iloc[0]

