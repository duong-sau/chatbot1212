import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import numpy
import time
import pandas as pd
from Static.Define import PathCommon

# if __name__ == '__main__':
#     df = pd.read_csv(PathCommon.learn_data)
#     df = df.sort_values("target")
#     count = df.target.value_counts()
#     count.to_csv('count.csv')


import socket
import matplotlib.pyplot as plt

HOST = '127.0.0.1'
PORT = 8000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

line = [50]

x_t5 = [0,10,20]
x_bm = [0]
y_t5 = [0,1,2]
y_bm = [0]
fig, ax = plt.subplots(2, 1)

line_t5, = ax[0].plot(x_t5, y_t5, 'r-')
ax[0].set_ylim(0, 10)
ax[0].set_xlim(0, 275)

line_bm, = ax[1].plot(x_bm, y_bm, 'r-')
ax[1].set_ylim(0, 10)
ax[1].set_xlim(0, 275)
ax[1].axvline(x=line[0])
plt.draw()
plt.pause(0.01)

plt.show()