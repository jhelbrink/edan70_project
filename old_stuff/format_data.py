import os
import json
from numpy import array
from keras.utils import np_utils
from keras import preprocessing
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


max_features = 10000
maxlen = 50

## The job we are building our model for
job_to_test = 'Q81096'

x = []
y = []

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
                x_word.append(word)
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


print('padding')
#x = preprocessing.sequence.pad_sequences(x, maxlen=maxlen)
pickle.dump(x , open('./good_stuff/x.p', 'wb'))
pickle.dump(y_test , open('./good_stuff/y.p', 'wb'))
pickle.dump(y , open('./good_stuff/y_multi.p', 'wb'))
