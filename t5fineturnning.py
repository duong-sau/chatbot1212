# -*- coding: utf-8 -*-
"""T5FineTurnning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-zJgQT_mJO5Asm9yPkkthVAv4_qTbAlY
"""

import torch
import pandas as pd
import numpy as np
from transformers import TrainingArguments, Trainer, AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers.optimization import AdamW, AdafactorSchedule
from torch.utils.data import Dataset

### Config
MODEL = {
    'name': 't5-small',
    'data_link': "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Cluster/IntentClassification/Positive/learn_data.csv",
    'num_freeze': [1, 1, 1, 0, 1, 0],
    'SEED': 42
}

strategy = 'epoch'
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/checkpoint",
    overwrite_output_dir=True,
    save_strategy=strategy,
    disable_tqdm=False,
    debug="underflow_overflow",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    evaluation_strategy='epoch',
    # logging_steps = 16,
    # eval_steps=16,
    fp16=False,
    warmup_steps=10,
    learning_rate=3e-5,
    adam_epsilon=1e-3,
    weight_decay=0.01,
    save_total_limit=10,
    load_best_model_at_end=True,
)


def getOptimizer(model):
    return AdamW(model.parameters(), lr=1e-3, relative_step=False, warmup_init=False)


def freezeLayer(model, freeze):
    for index, layer in enumerate(model.base_model.encoder.block):
        if freeze[index] == 0:
            print('not freeze layer:' + str(index + 1))
            continue
        elif freeze[index] == 1:
            print('freeze layer     :' + str(index + 1))
            for param in layer.parameters():
                param.requires_grad = False
        else:
            raise "freeze layer invalid format"


def tokenConfig(tokenizer):
    assert tokenizer
    tokenizer.padding_side = "left"


def train_validate_test_split(df, train_percent=.8):
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.iloc[perm[:train_end]]
    test = df.iloc[perm[train_end:]]
    return train, test


data = pd.read_csv(MODEL['data_link'], header=0)
data = data.astype(str)


class SiameseDataset(Dataset):
    def __init__(self, tokenizer, df, max_len=2048):
        self.data_column = [(lambda x: "classification: " + df.iloc[x]["sentence"] + '</s>')
                            (x) for x in range(len(df))]
        self.class_column = [(lambda x: df.iloc[x]["target"] + '</s>')
                             (x) for x in range(len(df))]
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_column)

    def __getitem__(self, index):
        tokenized_inputs = self.tokenizer.encode_plus(self.data_column[index], max_length=self.max_len,
                                                      padding='longest', return_tensors="pt")
        tokenized_targets = self.tokenizer.encode_plus(self.class_column[index], max_length=20, pad_to_max_length=True,
                                                       return_tensors="pt")
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()
        src_mask = tokenized_inputs["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask,
                "label": target_ids}


tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
tokenConfig(tokenizer=tokenizer)
assert tokenizer

config = T5Config.from_pretrained(MODEL['name'])
# config.num_decoder_layers = MODEL['num_decoder_layers']
model = T5ForConditionalGeneration.from_pretrained(MODEL['name'], config=config)
freezeLayer(model, MODEL['num_freeze'])
# optimizer = getOptimizer(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.cpu()
assert model

train_data, val_data = train_validate_test_split(data)
train_data = train_data[0:2]
val_data = val_data[0:10]
train_dataset = SiameseDataset(df=train_data, tokenizer=tokenizer)
val_dataset = SiameseDataset(df=val_data, tokenizer=tokenizer)
assert_data = train_dataset.__getitem__(1)
assert_inputs = assert_data['input_ids']
assert assert_inputs[-1] == 1
assert_label = assert_data['label']
assert assert_label[-1] == 1


def model_init():
    return T5ForConditionalGeneration.from_pretrained(MODEL['name'], config=config)


def get_labels(labels_str_ids):
    result = []
    for label_ids in labels_str_ids:
        r = []
        ss = label_ids.split()
        for s in ss:
            k = 0
            try:
                k = float(s)
            except ValueError:
                k = -1
            r.append(k)
        while len(r) < 3:
            r.append(-1)
        result.append(r[0:3])
    return torch.tensor(result)


def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    return np.mean(acc_list)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0].argmax(axis=-1)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    l0 = get_labels(labels_str_ids=labels_str)
    predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    p = get_labels(labels_str_ids=predictions_str)
    acc = hamming_score(y_pred=p, y_true=l0)
    return {
        'accuracy': acc,
        'f1': acc,
        'precision': acc,
        'recall': acc
    }

def hyperparameter_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.05, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-4, 1e-2, log=True)
    }


# lr_scheduler =  WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    model_init=model_init
)
trainer.hyperparameter_search(direction="maximize", hp_space=hyperparameter_space)
