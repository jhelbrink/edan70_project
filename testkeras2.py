import os
import json
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.datasets import imdb
from numpy import array

max_features = 10000
maxlen = 20

## The job we are building our model for
job_to_test = 'Q81096'

x = []
#y = []

## y for job to test, either 1 or 0
y_test = []

## DICT for word usage
word_list = {}

## EXTRACT WORDS AND COUNT HOW OFTEN THEY ARE USED
for filename in os.listdir('person_data2'):
    with open('person_data2/' + filename) as f:
        data = json.load(f)
        for person in data:
            words = person['first_paragraph'].split(' ')
            for word in words:
                if word in word_list:
                    word_list[word] = word_list[word] + 1
                else:
                    word_list[word] = 1

## THIS IS JUST FOR THE TEST CASE IN THE BOTTOM
sentence = 'Sir Timothy John "Tim" Berners-Lee, KBE, FRS, född 8 juni 1955 i London i England, är skaparen av World Wide Web, en teknik som skapade förutsättningar för en bredare användning av Internet, och chef för World Wide Web Consortium.'
for word in sentence:
    if word in word_list:
        word_list[word] = word_list[word] + 1
    else:
        word_list[word] = 1


common_words = list(reversed(sorted(word_list.items(), key=lambda item: item[1])))
for i, word in enumerate(common_words):
    common_words[i] = word[0]

#Push to x and y
for filename in os.listdir('person_data2'):
    with open('person_data2/' + filename) as f:
        data = json.load(f)
        for person in data:
            isJob = False
            words = person['first_paragraph'].split(' ')
            x_word = []
            for word in words:
                x_word.append(common_words.index(word))
            for job in person['jobs']:
                x.append(x_word)
                y.append(job)
                if(job==job_to_test):
                    isJob = True
                if(isJob):
                    y_test.append(1)
                else:
                    y_test.append(0)

x = array(x)
y_test = array(y_test)
print(x[0])
print(y[0])
print(max(x))
print(len(y))

print('padding')
x = preprocessing.sequence.pad_sequences(x, maxlen=maxlen)

print('Adding embedd')
model = Sequential()
model.add(Embedding(200000, 8, input_length=maxlen))

print('Adding Flatten')
model.add(Flatten())

print('Adding Dense')
model.add(Dense(1, activation='sigmoid'))

print('Compiling')
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['acc'])

print('Fitting')
history = model.fit(x, y_test,
epochs=4,
batch_size=32,
validation_split=0.2)

sentence = 'Sir Timothy John "Tim" Berners-Lee, KBE, FRS, född 8 juni 1955 i London i England, är skaparen av World Wide Web, en teknik som skapade förutsättningar för en bredare användning av Internet, och chef för World Wide Web Consortium.'
test = []
for word in sentence:
    if word in common_words:
        test.append(common_words.index(word))
test = array(test)
print(model.predict(test))
