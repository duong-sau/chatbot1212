import random
import torch
from torch.utils.data import Dataset
import pandas as pd

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}
MAX_LENGTH = 768  # {768, 1024, 1280, 1600}


class myDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data_df = data
        self.tokenizer = tokenizer
        # ---------------------------------------------#

    def __len__(self):
        return len(self.data_df[1, 1])

    # ---------------------------------------------#

    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        kw = self.join_keywords(keywords, self.randomize)

        input = SPECIAL_TOKENS['bos_token'] + self.title[i] + SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS[
            'sep_token'] + self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=MAX_LENGTH,
                                        padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}
