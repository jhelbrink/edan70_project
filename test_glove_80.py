import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense, Input, LSTM, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.callbacks import Callback, EarlyStopping
from keras.utils.np_utils import to_categorical
import json


x_train = pickle.load(open('./good_stuff/x_train.p', 'rb'))
y_train = pickle.load(open('./good_stuff/y_train.p', 'rb'))

x_test = pickle.load(open('./good_stuff/x_test.p', 'rb'))
y_test = pickle.load(open('./good_stuff/y_test.p', 'rb'))
print(y_train.shape)



embedding_matrix = pickle.load(open('./good_stuff/embedding_matrix.p', 'rb'))

embedding_dim = 100
maxlen = 150
max_words = 10000


"""
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4638, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['acc'])
history = model.fit(x_train, y_train,
epochs=5,
batch_size=250)
model.save_weights('./pre_trained_glove_model.h5')
"""

"""
inp = Input(shape=(maxlen,))
x = Embedding(max_words, embedding_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
x = GlobalMaxPool1D()(x)
x = Dense(60, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(4636, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
"""
"""
print(y_train.shape)
print(x_train.shape)
model = Sequential()
model.add(Embedding(max_words, 80, input_length=maxlen))
model.add(Bidirectional(LSTM(80, return_sequences=True, dropout=0.25, recurrent_dropout=0.25)))
model.add(GlobalMaxPool1D())
model.add(Dense(80, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1], activation="softmax"))
"""

model = Sequential()
model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix]))
model.add(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(512, input_shape=(max_words,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False


"""
inp = Input(shape=(maxlen,))
x = Embedding(max_words, embedding_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
x = GlobalMaxPool1D()(x)
x = Dense(60, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1371, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
"""


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1000, epochs=100, validation_split=0.05, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

model_json = model.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights('./model_weights.h5')
