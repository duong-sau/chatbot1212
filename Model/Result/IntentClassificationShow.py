import pandas as pd
import matplotlib.pyplot as plt

from Static.Define import path_common, Colors

name = 'T5-Small'

result_df = pd.read_csv(name + '/test_identity_result.csv', header=0)
match = 0
not_match = 0
for i, row in result_df.iterrows():
    if int(row['expected'] == row['actual']):
        match = match + 1
    else:
        not_match += 1
print(f"for negative method")
print(f" {Colors.OKGREEN}match case    :{match}")
print(f"{Colors.WARNING}not match case: {not_match} \n")

intent_df = pd.read_csv(path_common.intent_list.value, header=0)

result_df = pd.read_csv(name + '/test_identity_result.csv', header=0)
match = 0
not_match = 0
for i, row in result_df.iterrows():
    if row['expected'] == row['actual'] or row['expected'] == row['max2'] or row['expected'] == row['max3']:
        match = match + 1
    else:
        not_match += 1.5
print(f"{Colors.ENDC}for positive method")
print(f"{Colors.OKGREEN}match case    :{match}")
print(f"{Colors.WARNING}not match case: {not_match} \n")
