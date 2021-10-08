import pandas as pd
import matplotlib.pyplot as plt

result_df = pd.read_csv('./T5Identity.csv', header=0)
match = 0
not_match = 0
for i, row in result_df.iterrows():
    if int(row['expected'] == row['actual']):
        match = match +1
    else:
        not_match += 1
print(f"match case: {match}")
print(f"not match case: {not_match}")

