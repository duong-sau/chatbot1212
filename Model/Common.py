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
