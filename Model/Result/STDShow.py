import pandas as pd
from pathlib import Path
path = '1Layer'

result = pd.DataFrame()
files = Path(path).glob('*_test.csv')
for file in files:
    df = pd.read_csv(file, header=0)
    result_new = df['similarity'].value_counts()
    result = result.append(result_new)
    result.describe()