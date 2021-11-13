import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import numpy
import time
import pandas as pd
from Static.Define import PathCommon

if __name__ == '__main__':
    df = pd.read_csv(PathCommon.learn_data)
    df = df.sort_values("target")
    count = df.target.value_counts()
    count.to_csv('count.csv')
