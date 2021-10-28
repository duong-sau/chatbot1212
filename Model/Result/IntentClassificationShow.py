import pandas as pd
import numpy as np
from Static.Define import path_common, Colors
from matplotlib.pyplot import plot
import matplotlib
"""
layer = [1, 2, 3, 4, 5, 6]
y = np.empty(shape=[1, 2])
for x in layer:
    name = "freeze" + str(x)
    print('number of freeze:', name, '==========')
    result_df = pd.read_csv(name + '/test_identity_result.csv', header=0)
    match = 0
    not_match = 0
    for i, row in result_df.iterrows():
        if int(row['expected'] == row['actual']):
            match = match + 1
        else:
            not_match += 1
    # print(f"for negative method")
    print(f" {Colors.OKGREEN}match case    :{match}")
    print(f"{Colors.WARNING}not match case: {not_match} \n")
    n = np.array([[x, match]])
    y = np.append(y, n, axis=0)
    print('==================================')
y = np.delete(y, 0, 0)
y[0:, 1] = y[0:, 1] / 47*100
plot(y)
matplotlib.pyplot.show()
"""
result_df = pd.read_csv('./A2/test_identity_result.csv', header=0)
intent_df = pd.read_csv(path_common.intent_list.value, header=0)
match = 0
not_match = 0
for i, row in result_df.iterrows():
    if row['expected'] == row['max1']:
        match = match + 1
    else:
        not_match += 1
print(f"{Colors.ENDC}for positive method")
print(f"{Colors.OKGREEN}match case    :{match}")
print(f"{Colors.WARNING}not match case: {not_match} \n")
