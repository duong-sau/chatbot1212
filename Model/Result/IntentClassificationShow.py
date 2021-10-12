import pandas as pd
import matplotlib.pyplot as plt

from Static.Define import path_common, Colors

result_df = pd.read_csv('./positive-freeze/test_identity_result.csv', header=0)
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

result_df = pd.read_csv('./positive-freeze/test_identity_result.csv', header=0)
match = 0
not_match = 0
for i, row in result_df.iterrows():
    if row['expected'] == row['actual']:
        match = match + 1
    elif intent_df.loc[intent_df['intent_index'] == row['expected']].iloc[0]['intent_group_index'] == \
            intent_df.loc[intent_df['intent_index'] == row['actual']].iloc[0]['intent_group_index']:
        not_match += 0.75
    else:
        not_match += 1.5
print(f"{Colors.ENDC}for positive method")
print(f"{Colors.OKGREEN}match case    :{match}")
print(f"{Colors.WARNING}not match case: {not_match} \n")
