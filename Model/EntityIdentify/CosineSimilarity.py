import pandas as pd
from tqdm.auto import tqdm
from os import path, mkdir
from FlaskDeploy.Cosine import get_index_sbert
import socket

HOST = '127.0.0.1'
PORT = 8000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
print('connecting to %s port ' + str(server_address))
s.connect(server_address)
tqdm.pandas()

result_dir = '../Result/SBert'
result_path = result_dir + '/result.csv'
if not path.exists(result_dir):
    mkdir(result_dir)

test_link = "D:\\chatbot1212\\Model\\Data\\STSB\\test.csv"

test_df = pd.read_csv(test_link, header=0)
columns = ["test_id", "expected", "actual", "max2", "max3"]
result_df = pd.DataFrame(columns=columns)

group = []
for i in range(30):
    group.append(float(i))

for index, row in tqdm(test_df.iterrows(), leave=False, total=len(test_df)):
    test_sentence = row['sentence']

    max_list = get_index_sbert(test_sentence, group=group, top_k=3, s=s)[0]
    new_row = {'test_id': row["sentence_index"], 'expected': row["label_index"], 'actual': max_list[0],
               'max2': max_list[1], 'max3': max_list[2]}
    result_df = result_df.append(new_row, ignore_index=True)
result_df.to_csv(path_or_buf=result_path, mode='w', index=False)
