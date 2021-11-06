# import
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForSequenceClassification
from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from FlaskDeploy.BM25 import get_index_bm25
from FlaskDeploy.Cosine import get_index_bert
from Static.Answer import get_index, pandas_to_json, get_cluster
from Static.Config import tokenizer_config, get_device, MODEL

tqdm.pandas()

device = get_device()

# import model
siamese_tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
tokenizer_config(tokenizer=siamese_tokenizer)
siamese_model = T5ForConditionalGeneration.from_pretrained('Model/CheckPoint/CommandRefrence')
siamese_model.to(device)
print('load siamese model success .to(' + str(device.type) + ')')

answer_df = pd.read_csv('https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/Mining/answer_list'
                        '.csv')


def group_answer(question, t5_top_p):
    group = get_cluster(question, t5_top_p)
    return group


def get_answer(index_and_highlight):
    index, highlight = index_and_highlight
    r = pd.DataFrame()
    for i, idx in enumerate(index):
        row = answer_df[answer_df['label_index'] == idx].iloc[0]
        new = {'label': row['label'], 'answer': row['answer'], 'first': row['first'], 'highlight': highlight[i]}
        r = r.append(new, ignore_index=True)
    return pandas_to_json(answer_df=r)


def answer_t5(question, top_k, group):
    index = get_index(question, group, top_k, siamese_tokenizer=siamese_tokenizer, siamese_model=siamese_model)
    answer = get_answer(index)
    return answer


def answer_cosine(question, top_k, group):
    index = get_index_bert(question, group, top_k)
    answer = get_answer(index)
    return answer


def answer_bm25(question, top_k):
    index = get_index_bm25(question, top_k)
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
                answer.ans = t5_answer
            elif method == 'Cosine':
                if parameter == 'BM25':
                    cosine_answer = answer_bm25(question=question, top_k=top_k)
                else:
                    group = group_answer(question, 2)
                    cosine_answer = answer_cosine(question=question, top_k=top_k, group=group)
                answer.ans = cosine_answer
        response = jsonify({'answer': answer})
        return response
    else:
        return "<h1>Error occurred<h1>"


app.run()
