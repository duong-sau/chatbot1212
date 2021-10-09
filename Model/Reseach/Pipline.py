import pandas as pd

from Model.Reseach.T5Base import base_start
from Model.Reseach.T5Large import large_start
from Model.Reseach.T5Small import small_start

if __name__ == '__main__':
    df = pd.DataFrame(columns=['do_sample', 'elapsed_time', 'fast_token', 'length', 'model_name', 'padding', 'result'])
    df.to_csv('elapsed_time.csv', mode='w', index=False)
    small_start()
    base_start()
    large_start()
