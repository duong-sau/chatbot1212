import pandas as pd
from Static.Define import PathCommon

result_df = pd.read_csv('../Result/CommandRefrence/result.csv', header=0)
sentence_df = pd.read_csv(PathCommon.sentence_list)
join = pd.merge(left=result_df, right=sentence_df, left_on='test_id', right_on='sentence_index')
join.head()
match = join[join['expected'] == join['actual']]
not_match = join[join['expected'] != join['actual']]
