# Config
# import os
# import random
# import numpy as np
import torch
# from transformers import TrainingArguments

MODEL = {
    'name': 't5-small',
    'classification_name': 'bert-base-uncased',
    'data_link': "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/POS"
                 "/learn_data.csv",
    'max_length': 512,
    'num_class': 2,
    'num_decoder_layers': 6,
    'num_freeze': [2, 3, 4],
    'SEED': 1211
}


# strategy = 'epoch'
# training_args = TrainingArguments(
#     output_dir="/content/drive/MyDrive",
#     overwrite_output_dir=True,
#     save_strategy=strategy,
#     disable_tqdm=False,
#     debug="underflow_overflow",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     gradient_accumulation_steps=16,
#     evaluation_strategy=strategy,
#     # logging_steps = 16,
#     # eval_steps=16,
#     fp16=False,
#     warmup_steps=100,
#     learning_rate=1e-3,
#     adam_epsilon=1e-3,
#     weight_decay=0.01,
#     save_total_limit=1,
#     load_best_model_at_end=False,
# )
#
#
# # def get_optimizer(model):
# #     return Adafactor(model.parameters(), lr=1e-3, relative_step=False, warmup_init=False)
#
#
# def freeze_layer(model, freeze):
#     for index, layer in enumerate(model.base_model.encoder.block):
#         if index + 1 not in freeze:
#             print('not freeze layer:' + str(index + 1))
#             continue
#         else:
#             print('freeze layer     :' + str(index + 1))
#             for param in layer.parameters():
#                 param.requires_grad = False
#
#
def tokenizer_config(tokenizer):
    assert tokenizer
    tokenizer.padding_side = "left"
#
#
# def seed():
#     device = get_device()
#     torch.device(device=device)
#     random.seed(MODEL['SEED'])
#     os.environ['PYTHONHASHSEED'] = str(MODEL['SEED'])
#     np.random.seed(MODEL['SEED'])
#     torch.manual_seed(MODEL['SEED'])
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#
#
# def train_validate_test_split(df, train_percent=.8):
#     seed()
#     perm = np.random.permutation(df.index)
#     m = len(df.index)
#     train_end = int(train_percent * m)
#     train = df.iloc[perm[:train_end]]
#     test = df.iloc[perm[train_end:]]
#     return train, test
#
#
def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
