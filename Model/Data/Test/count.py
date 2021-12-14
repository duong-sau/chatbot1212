import pandas as pd
test = pd.read_csv('/test.csv', header=0)
test.value_counts()