import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense, Input, LSTM, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.utils.np_utils import to_categorical
import json

print('What model do you want to build?')
print('1. 50-500')
print('2. 100-500')
dataset = input()
dir_to_load_from = ''
model_to_save = ''
model_weights_to_save = ''

if dataset == '1':
    print('Building the larger model')
    min_jobs = 50
    dir_to_load_from = './50-500-data/'
    model_to_save = './models/model_in_json_big.json'
    model_weights_to_save = './models/model_weights_big.h5'
elif dataset == '2':
    print('Building the smaller model')
    min_jobs = 100
    dir_to_load_from = './100-500-data/'
    model_to_save = './models/model_in_json_small.json'
    model_weights_to_save = './models/model_weights_small.h5'
else:
    sys.exit()

x_train = pickle.load(open(dir_to_load_from + 'x_train.p', 'rb'))
y_train = pickle.load(open(dir_to_load_from + 'y_train.p', 'rb'))

x_test = pickle.load(open(dir_to_load_from + 'x_test.p', 'rb'))
y_test = pickle.load(open(dir_to_load_from + 'y_test.p', 'rb'))

embedding_matrix = pickle.load(open(dir_to_load_from + 'embedding_matrix.p', 'rb'))

embedding_dim = 100
maxlen = 150
max_words = 10000

model = Sequential()
model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix]))
model.add(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(512, input_shape=(max_words,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1]))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=120, epochs=5, validation_split=0.15)

model_json = model.to_json()
with open(model_to_save, "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights(model_weights_to_save)
