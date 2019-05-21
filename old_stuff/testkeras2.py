import os
import json
import pandas as pd
from keras import preprocessing
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.datasets import imdb
from numpy import array
from keras.utils.np_utils import to_categorical
import pickle

maxlen = 80

## The job we are building our model for
job_to_test = 'Q81096'
"""
x = pickle.load( open( "./good_stuff/x.p", "rb" ) ) #(43716, 50)
print(x)
y = pickle.load( open( "./good_stuff/y.p", "rb" ) ) #(43716,)
y_multi = pickle.load( open( "./good_stuff/y_multi.p", "rb" ) )
print(len(y_multi))
"""
x = pickle.load(open('./good_stuff/x_train.p', 'rb'))
y_multi = pickle.load(open('./good_stuff/y_train.p', 'rb'))
y_multi = to_categorical(y_multi)
print(y_multi.shape)
#x = x.reshape(1, 43716, 1026)
"""
class_list = {}
classes_sorted = []

for c in y_multi:
    if c in class_list:
        class_list[c] = class_list[c] + 1
    else:
        class_list[c] = 1
classes_sorted = list(reversed(sorted(class_list.items(), key=lambda item: item[1])))
print(max(classes_sorted))

for i, word in enumerate(classes_sorted):
    classes_sorted[i] = word[0]

for i, c in enumerate(y_multi):
    y_multi[i] = classes_sorted.index(c)
#print(y_multi)
print(max(y_multi))
y_multi = to_categorical(y_multi)
"""

##FIRST TEST WE SHOWED PIERRE
"""
print('Adding embedd')
model = Sequential()
model.add(Embedding(200000, 8, input_length=maxlen)) #Embedds the word for a vector representation.

print('Adding Flatten')
model.add(Flatten())

print('Adding Dense')
model.add(Dense(1, activation='sigmoid'))

print('Compiling')
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['acc'])
"""


## FEED FORWARD
"""
model = Sequential([
Dense(100),
Activation('relu'),
Dense(1),
Activation('sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
"""
## LSTM
"""
model = Sequential()
model.add(Embedding(60000, 8, input_length=maxlen))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
"""

"""
## FEED FORWARD MULTI CLASS, FAILED
model = Sequential([
Dense(500, input_shape=(43716,1026)),
Activation('relu'),
Dense(500),
Activation('relu'),
Dense(10),
Activation('softmax')
])
optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
"""

## MULTI CLASS
#inputs = Input(shape=(80, ))
model = Sequential()
model.add(Embedding(25000, 100, input_length=80))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(4638, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['acc'])

print('Fitting')
history = model.fit(x, y_multi,
epochs=20,
batch_size=5000,
verbose=1,
validation_split=0.2)
