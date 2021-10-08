import os

from bert_embedding import BertEmbedding
from keras import Input, Model
from keras.layers import LSTM, Subtract, Multiply, Concatenate, Dense, Dropout
from torch.optim import Adam
from keras import backend as K
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def getEM(sentence: str):
    sentences = sentence.split('\n')
    bert_embedding = BertEmbedding()
    result = bert_embedding(sentences)
    return result[1]



input_1 = Input(shape=train_q1_seq.shape[1])
input_2 = Input(shape=train_q2_seq.shape[1])
common_LSTM = LSTM(64, return_sequence=True, activation='relu')
lstm_1 = getEM("1")
lstm_2 = getEM("2")
vector_1 = common_lstm(lstm_1)
vector_1 = Flatten()(vector_1)

vector_2 = common_lstm(lstm_2)
vector_2 = Flatten()(vector_2)

x3=Subtract()[vector_1, vector_2]
x3 = Multiply()[x3, x3]
x1 = Multiply()[vector_1, vector_1]
x2 = Multiply()[vector_2, vector_2]
x4 = Subtract()[x1, x2]

x5 = lambda (cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])
conc = Concatenate(axis=-1)[x5, x4, x3]

x = Dense(100, activation='relu', name='convolution_layer')(conc)
x= Dropout(0.01)(x)
out = Dense(1, activation='sigmoid', name='out')

model = Model([input_1, input_2], out)
model.compile(loss="binary_crossentropy", metrics=['acc', auroc], optimizer=Adam(0.0001))

model.summary()

checkpoint_path = "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit([train_q1_seq,train_q2_seq],y_train.values.reshape(-1,1), epochs = 5,
          batch_size=64,validation_data=([val_q1_seq, val_q2_seq],y_val.values.reshape(-1,1)),  callbacks=[cp_callback])

