# define
import os
import random

import numpy as np
import torch

SEED = 1211


def toT5sentence(sentence1, sentence2):
    prefix = 'stsb '
    s1 = 'sentence1: '
    s2 = '. sentence2: '
    T5Sentence = prefix + s1 + sentence1 + s2 + sentence2
    return T5Sentence


def getSimilarity(tokenizer, model, test_sentence, compare_sentences):
    T5_format_sentence = toT5sentence(sentence1=test_sentence, sentence2=compare_sentences)
    inputs = tokenizer(T5_format_sentence, return_tensors="pt", padding=True)
    output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                      do_sample=False)
    similarity = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return float(similarity[0])


def seed():
    torch.device("cuda")
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_validate_test_split(df, train_percent=.8):
    seed()
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.iloc[perm[:train_end]]
    test = df.iloc[perm[train_end:]]
    return train, test
