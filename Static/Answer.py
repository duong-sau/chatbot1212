# group classification
import pandas as pd
from tqdm import tqdm

from Model.Common import getSimilarity


def get_class(text, class_model, class_tokenizer):
    inputs = class_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = class_model(**inputs)
    probability = outputs[0].softmax(1)
    return [1.0, 2.0][probability.argmax()]


# answer the question
def get_index(question, group, siamese_tokenizer, siamese_model):
    result_df = pd.read_csv(
        "https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data/IntentClassification/sentence_list"
        ".csv",
        header=0)
    result_df = result_df[result_df['intent_group_index'] == group]
    for i, r in tqdm(result_df.iterrows()):
        compare_sentences = r["sentence"]
        similarity = getSimilarity(tokenizer=siamese_tokenizer, model=siamese_model, test_sentence=question,
                                   compare_sentences=compare_sentences)
        result_df.loc[i, "similarity"] = similarity
    result_df['similarity'] = pd.to_numeric(result_df['similarity'], errors='coerce')
    mean_df = result_df.groupby(["intent_index"])["similarity"].mean().reset_index().sort_values("similarity")
    max = []
    try:
        max = mean_df.iloc[-5:]['intent_index'].tolist()
    except:
        index = -1
    return max


def pandas_to_json(answer_df):
    js = answer_df.to_json(orient='columns')
    return js
