import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from FlaskDeploy.BERT import get_index_bert
from FlaskDeploy.BM25 import get_index_bm25
from FlaskDeploy.Cosine import get_index_sbert
from FlaskDeploy.T5 import get_index_t5, get_cluster

import socket

HOST = '127.0.0.1'
PORT = 8000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
print('connecting to %s port ' + str(server_address))
s.connect(server_address)

answer_df = pd.read_csv('D:\\chatbot1212\\Model\\Data\\STSB\\answer_list.csv')


def pandas_to_json(answers):
    js = answers.to_json(orient='index')
    return js


def group_answer(question, t5_top_p):
    group = get_cluster(question, t5_top_p)
    return group


def get_answer(index_and_highlight):
    index, highlight = index_and_highlight
    r = pd.DataFrame()
    for i, idx in enumerate(index):
        row = answer_df[answer_df['label_index'] == idx].iloc[0]
        new = {'label': row['label'], 'answer': row['answer'], 'first': row['first'], 'highlight': highlight[i]}
        # new = {'label': row['label'], 'answer': row['answer'], 'first': row['first'], 'highlight': ""}
        r = r.append(new, ignore_index=True)
    return pandas_to_json(answers=r)

direct_df = pd.read_csv(
    "https://raw.githubusercontent.com/duong-sau/chatbot1212/1c1bf2900a2e319d28f91ab6e313df2e9bfc3258/Model"
    "/DirectClassification/IntentClassification/answer_list.csv")


def get_answer_direct(index_and_highlight):
    index, highlight = index_and_highlight
    r = pd.DataFrame()
    for i, idx in enumerate(index):
        row = direct_df[direct_df['label_index'] == idx].iloc[0]
        # new = {'label': row['label'], 'answer': row['answer'], 'first': row['first'], 'highlight': highlight[i]}
        new = {'label': row['label'], 'answer': row['answer'], 'first': row['first'], 'highlight': ""}
        r = r.append(new, ignore_index=True)
    return pandas_to_json(answers=r)


def answer_t5(question, top_k, group):
    index = get_index_t5(question, group, top_k, s)
    answer = get_answer_direct(index)
    return answer


def answer_cosine(question, top_k, group):
    index = get_index_sbert(question, group, top_k, s)
    answer = get_answer(index)
    return answer


def answer_bm25(question, top_k):
    index = get_index_bm25(question, top_k, s)
    answer = get_answer(index)
    return answer

def answer_bert(question, top_k):
    index = get_index_bert(question, top_k, s)
    answer = get_answer(index)
    return answer

# run app
app = Flask(__name__)
# run_with_ngrok(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def home():
    s.sendall(bytes("index", "utf8"))
    return "<h1>Welcome to iqtree chatbot server!</h1>"


@app.route('/question', methods=['GET'])
@cross_origin()
def login():
    if request.method == 'GET':
        question = request.args.get('question')
        left_method = request.args.get('left_method')
        right_method = request.args.get('right_method')
        left_parameter = request.args.get('left_parameter')
        right_parameter = request.args.get('right_parameter')

        answer = {}
        top_k = 1
        try:
            top_k = int(request.args.get('top_k'))
        except ValueError:
            pass
        for method, parameter, ans in [(left_method, left_parameter, 'left_answer'),
                                       (right_method, right_parameter, 'right_answer')]:
            if method == 'T5':
                t5_top_p = 1
                try:
                    t5_top_p = int(parameter)
                except ValueError:
                    pass
                group = group_answer(question, t5_top_p)
                t5_answer = answer_t5(question=question, top_k=top_k, group=group)
                answer[ans] = t5_answer
            elif method == 'Cosine':
                if parameter == 'BM25':
                    cosine_answer = answer_bm25(question=question, top_k=top_k)
                else:
                    group = group_answer(question, 2)
                    cosine_answer = answer_cosine(question=question, top_k=top_k, group=group)
                answer[ans] = cosine_answer
            elif method == 'bert':
                bert_answer = answer_bert(question=question, top_k=top_k)
                answer[ans] = bert_answer
        response = jsonify({'answer': answer})
        return response
    else:
        return "<h1>Error occurred<h1>"


app.run()
