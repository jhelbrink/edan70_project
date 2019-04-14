import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import os
import json

x = []
y = []
for filename in os.listdir('person_data2'):
    with open('person_data2/' + filename) as f:
        data = json.load(f)
        for person in data:
            for job in person['jobs']:
                x.append(person['first_paragraph'])
                y.append(job)
print(len(x))
num_classes = len(list(set(y)))


tokenizer = Tokenizer(num_words=1000)

for i, xs in enumerate(x):
    x[i] = one_hot(xs, round(len(xs)*1.3))
for i, ys in enumerate(y):
    y[i] = one_hot(ys, round(len((y))))

x = tokenizer.sequences_to_matrix(x, mode='binary')
y = keras.utils.to_categorical(y, len(y))

model = Sequential()
model.add(Dense(512,input_shape=(1000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.metrics_names)

batch_size = 128
epochs = 5

history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.1)
score = model.evaluate(x,y,batch_size=batch_size, verbose=1)

print('Test loss: {}'.format(score[0]))
print('Test acc: {}'.format(score[1]))
