import pandas as pd
pos = pd.read_csv('./Positive/learn_data.csv')
neg = pd.read_csv("./Negative/learn_data.csv")
print(pos.target.value_counts())
print(neg.target.value_counts())