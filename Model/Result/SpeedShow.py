import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_df = pd.read_csv('elapsed_time.csv', header=0)
data_df = data_df.groupby(['model_name', 'do_sample', 'fast_token'])

b11 = data_df.get_group(('t5-base', '1.0', '1.0')).reset_index()
p11 = b11[['length', 'elapsed_time']].set_index('length').plot()

b00 = data_df.get_group(('t5-base', '1.0', '0.0')).reset_index()
p00 = b00[['length', 'elapsed_time']].set_index('length').plot()

s00 = data_df.get_group(('t5-small', '1.0', '0.0')).reset_index()
ps00 = s00[['length', 'elapsed_time']].set_index('length').plot()

plt.show()

#
# for key, item in data_df:
#     #a_group = data_df.get_group(key)
#     #print(a_group.head(), "\n")
#     print(key)
#     print(item)
