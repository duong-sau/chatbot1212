# import
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForSequenceClassification
from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from Static.Answer import get_class, get_index, pandas_to_json
from Static.Config import tokenizer_config, get_device, MODEL

tqdm.pandas()

device = get_device()

# import model
siamese_tokenizer = AutoTokenizer.from_pretrained(MODEL['name'])
tokenizer_config(tokenizer=siamese_tokenizer)
siamese_model = T5ForConditionalGeneration.from_pretrained('../Model/CheckPoint/111000')
siamese_model.to(device)
print('load siamese model success .to(' + str(device.type) + ')')

class_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
class_model = BertForSequenceClassification.from_pretrained('../Model/CheckPoint/Class')
class_model.to(device)
print('load classification model success .to(' + str(device.type) + ')')

answer_df = pd.read_csv('https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/Mining/answer_list'
                        '.csv')


def group_answer(question):
    group = get_class(question, class_model=class_model, class_tokenizer=class_tokenizer)
    return float(group)


def get_answer(index):
    r = answer_df[answer_df['intent_index'].isin(index)]
    r.head()
    answer = r['answer']
    return pandas_to_json(answer_df=answer)


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
        group = group_answer(question)
        index = get_index(question, group, siamese_tokenizer=siamese_tokenizer, siamese_model=siamese_model)
        answer = get_answer(index)
        response = jsonify({'answer': answer})
        return response
    else:
        return "<h1>Error occurred<h1>"


app.run()
