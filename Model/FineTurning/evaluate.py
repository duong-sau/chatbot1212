import os
import pandas as pd
from matplotlib import pyplot as plt

from Static.Define import Colors
from matplotlib.pyplot import plot
import matplotlib


def all_evaluate():
    out_df = pd.DataFrame()
    for item in os.listdir('../Result'):
        p = '../Result/' + item + "/result.csv"
        if not os.path.exists(p):
            continue
        result_df = pd.read_csv(p)
        conut = len(result_df)
        match_max1 = 0
        match_max2 = 0
        match_max3 = 0
        not_match = 0
        for i, row in result_df.iterrows():
            if row['expected'] == row['actual']:
                match_max1 = match_max1 + 1
            if row['expected'] == row['actual'] or row['expected'] == row['max2']:
                match_max2 = match_max2 + 1
            if row['expected'] == row['actual'] or row['expected'] == row['max2'] or row['expected'] == row['max3']:
                match_max3 = match_max3 + 1
            else:
                not_match += 1
        name = item
        try:
            name = int(name, 2)
            continue
        except ValueError:
            print(name)
        new_row = {'name': name, 'match_1': match_max1/conut, 'match_2': match_max2/conut, 'match_3': match_max3/conut}
        out_df = out_df.append(new_row, ignore_index=True)
    out_df.to_csv('../Result/all_result.csv', index=False)


def visualizer():
    result = pd.read_csv('../Result/all_result.csv', header=0)
    ax = plt.subplot2grid((1, 1), (0, 0))
    plt.xticks(label=result['name'], fontsize=12, rotation=30)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(40, 10.5)
    ax.bar(result['name'], result['match_1'])
    plt.show()
    fig.savefig('test2png.png', dpi=100)


def evaluate(name):
    result_df = pd.read_csv('../Result/' + name + '/result.csv', header=0)
    match_max1 = 0
    match_max2 = 0
    match_max3 = 0
    not_match = 0
    for i, row in result_df.iterrows():
        if row['expected'] == row['actual']:
            match_max1 = match_max1 + 1
        if row['expected'] == row['actual'] or row['expected'] == row['max2']:
            match_max2 = match_max2 + 1
        if row['expected'] == row['actual'] or row['expected'] == row['max2'] or row['expected'] == row['max3']:
            match_max3 = match_max3 + 1
        else:
            not_match += 1
    print(f"{Colors.OK_GREEN}match_max1 case    :{match_max1}")
    print(f"{Colors.OK_GREEN}match_max2 case    :{match_max2}")
    print(f"{Colors.OK_GREEN}match_max3 case    :{match_max3}")
    print(f"{Colors.WARNING}not match case     :{not_match}\n")


if __name__ == '__main__':
    all_evaluate()
    visualizer()
