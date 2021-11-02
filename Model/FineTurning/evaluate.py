import pandas as pd
import numpy as np
from Static.Define import PathCommon, Colors
from matplotlib.pyplot import plot
import matplotlib


def gird_evaluate():
    layer = [1, 2, 3, 4, 5, 6]
    y = np.empty(shape=[1, 2])
    for x in layer:
        name = "./freeze'" + str(x)
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
        print(f" {Colors.OK_GREEN}match case    :{match}")
        print(f"{Colors.WARNING}not match case: {not_match} \n")
        n = np.array([[x, match]])
        y = np.append(y, n, axis=0)
        print('==================================')
    y = np.delete(y, 0, 0)
    y[0:, 1] = y[0:, 1] / 47 * 100
    plot(y)
    matplotlib.pyplot.show()


def evaluate(name):
    result_df = pd.read_csv('../Result/' + name + '/result.csv', header=0)
    match = 0
    not_match = 0
    for i, row in result_df.iterrows():
        if row['expected'] == row['actual']:
            match = match + 1
        else:
            not_match += 1
    print(f"{Colors.END}for positive method")
    print(f"{Colors.OK_GREEN}match case    :{match}")
    print(f"{Colors.WARNING}not match case: {not_match} \n")


if __name__ == '__main__':
    evaluate('Label')
