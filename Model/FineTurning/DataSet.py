import torch
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, tokenizer, df, max_len=512):
        self.data_column = df["source"].values + '</s>'
        self.class_column = df['target'].values + '</s>'
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_column)

    def __getitem__(self, index):
        tokenized_inputs = self.tokenizer.encode_plus(self.data_column[index], max_length=self.max_len,
                                                      padding='longest', return_tensors="pt")
        tokenized_targets = self.tokenizer.encode_plus(self.class_column[index], max_length=4, pad_to_max_length=True,
                                                       return_tensors="pt")
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()
        src_mask = tokenized_inputs["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "label": target_ids}


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, df, max_len=512):
        self.data_column = df["source"].values
        self.class_column = df['target'].values
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_column)

    def __getitem__(self, index):
        tokenized_inputs = self.tokenizer.encode_plus(self.data_column[index], max_length=self.max_len,
                                                      truncation=True, padding=True, return_tensors="pt")
        source_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        token_type_ids = tokenized_inputs['token_type_ids']
        label = torch.tensor([self.class_column[index]])
        return {'source_ids': source_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'label': label}
