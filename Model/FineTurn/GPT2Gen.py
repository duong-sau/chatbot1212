import os
import io
import requests
import numpy as np
import pandas as pd
import re
import zipfile
import random
import time
import csv
import datetime
from itertools import compress
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    AdamW, get_linear_schedule_with_warmup, \
    TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
    RandomSampler, SequentialSampler
from IPython.display import clear_output
from Model.Data.Connecter.CSVConnecter import readNews, read_keywords, split_data
from Model.TextGenaretion.GPT2.DataSet.DataSetLoader import myDataset, MAX_LENGTH
from Static.Define import path_common


def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)  # GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer


def get_model(tokenizer, special_tokens=None, load_model_path=None):
    # GPT2LMHeadModel
    cpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    if special_tokens:
        print("model 3")
        config = AutoConfig.from_pretrained(MODEL,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)

    else:
        print("moldel4")
        config = AutoConfig.from_pretrained(MODEL,
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)

        # ----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)
    model = model.to(cpu)
    if special_tokens:
        # Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
        model.cuda()
    return model


def seed_everything(seed):
    torch.device("cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.zeros.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    # print(f"PyTorch version: {torch.__version__}")
    DEBUG = False

    INPUT_DIR = 'articles'

    USE_APEX = False
    APEX_OPT_LEVEL = 'O1'

    MODEL = 'gpt2'  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}

    UNFREEZE_LAST_N = 6  # The last N layers to unfreeze for training

    if USE_APEX:
        TRAIN_BATCHSIZE = 4
        BATCH_UPDATE = 16
    else:
        TRAIN_BATCHSIZE = 2
        BATCH_UPDATE = 32

    EPOCHS = 4
    LR = 5e-3
    EPS = 1e-8
    WARMUP_STEPS = 1e2

    SEED = 2020

    seed_everything(SEED)
    df = readNews()

    data = dict()
    for root, dirs, files in os.walk(path_common.article.value, topdown=True):
        t0 = time.time()

        for i, f in enumerate(files):
            # id, category, title, keywords, text
            id = int(f[:-4])
            tmp = df[['CATEGORY', 'TITLE']][df.ID == id].values
            category, title = tmp[0][0], tmp[0][1]

            with open(f'{path_common.article.value}/{f}', "r", encoding='cp932', errors='ignore') as infile:
                text = infile.read()

            data[id] = [title, text]

            if i % 1000 == 0 and i > 0:
                clear_output(wait=True)
                print(f"({os.getpid()}) Items processed: {i :,}/{len(files):,}; {(time.time() - t0) / 60 :.1f} minutes")

                if DEBUG:
                    break

    print(f"Number of articles: {len(data) :,}")
    keywords = read_keywords()

    all_keywords = set()
    for k, v in keywords.items():
        for w in v:
            all_keywords.add(w)

    for id in data.keys():
        data[id].append(keywords[id])

    print(f"Number of unique keywords: {len(all_keywords) :,}")
    tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
    print("model2")
    model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS)  # load_model_path='pytorch_model.bin')
    print("model1")
    # -------------------------------
    # - Freeze selective layers:
    # - Freeze all layers except last n:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        # Only un-freeze the last n transformer blocks
        if i + 1 > 12 - UNFREEZE_LAST_N:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True
    train_data, val_data = split_data(data)
    train_dataset = myDataset(train_data, tokenizer)
    val_dataset = myDataset(val_data, tokenizer, randomize=False)

    f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing'

    training_args = TrainingArguments(
        output_dir="/content/",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCHSIZE,
        per_device_eval_batch_size=TRAIN_BATCHSIZE,
        gradient_accumulation_steps=BATCH_UPDATE,
        evaluation_strategy="epoch",
        fp16=False,
        fp16_opt_level=APEX_OPT_LEVEL,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        adam_epsilon=EPS,
        weight_decay=0.01,
        save_total_limit=1,
        #load_best_model_at_end=False,
    )

    # ---------------------------------------------------#
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # ---------------------------------------------------#
    trainer.train()
    trainer.save_model()
