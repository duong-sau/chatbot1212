# import
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForSequenceClassification
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from FlaskDeploy.BM25 import get_index_bm25
from FlaskDeploy.Cosine import get_index_bert
from Static.Answer import get_class, get_index, pandas_to_json
from Static.Config import tokenizer_config, get_device, MODEL

tqdm.pandas()

device = get_device()

# import model
siamese_tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
tokenizer_config(tokenizer=siamese_tokenizer)
siamese_model = T5ForConditionalGeneration.from_pretrained('../Model/CheckPoint/CommandRefrence')
siamese_model.to(device)
print('load siamese model success .to(' + str(device.type) + ')')

answer_df = pd.read_csv('https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/Mining/answer_list'
                        '.csv')


def group_answer(question, t5_top_p):
    group = get_class(question, t5_top_p, class_model=siamese_model, class_tokenizer=siamese_tokenizer)
    return group


def get_answer(index_and_highlight):
    index, highlight = index_and_highlight
    r = pd.DataFrame()
    for i, idx in enumerate(index):
        row = answer_df[answer_df['intent_index'] == idx].iloc[0]
        new = {'intent': row['intent'], 'answer': row['answer'], 'first': row['first'], 'highlight': highlight[i]}
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
        top_k = 5
        try:
            top_k = int(request.args.get('top_k'))
        except ValueError:
            pass

        t5_top_p = 2
        try:
            t5_top_p = int(request.args.get('t5_top_p'))
        except ValueError:
            pass

        cosine_embedding = request.args.get('cosine_embedding')

        group = group_answer(question, t5_top_p)
        t5_answer = answer_t5(question=question, top_k=top_k, group=group)

        if cosine_embedding == 'bm25':
            cosine_answer = answer_bm25(question=question, top_k=top_k)
        else:
            cosine_answer = answer_cosine(question=question, top_k=top_k, group=group)

        response = jsonify({'answer': {'t5_answer': t5_answer, 'cosine_answer': cosine_answer}})
        return response
    else:
        return "<h1>Error occurred<h1>"


app.run()
