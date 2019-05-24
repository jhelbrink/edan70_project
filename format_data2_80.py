import os
import json
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from keras.utils import to_categorical
labelencoder = LabelEncoder()
one_hot = MultiLabelBinarizer()


glove_dir = './'

embedding_dim = 100
maxlen = 150
max_words = 10000

train = 240000
test = 50000

nr_labels = 0 #CAN BE USED IF WE WANT TO HAVE A SPECIFIC AMOUNT OF LABELS
nr_labels_to_have = 50 #NUMBER OF OCCURENCES OF OCCUPATION TO INCLUDE

texts = []
labels = []
persons = []

cur_label = 1

label_translate = {}

jobs_dict = {}

## EXTRACT WORDS AND COUNT HOW OFTEN THEY ARE USED
for filename in os.listdir('person_data2'):
    with open('person_data2/' + filename, encoding="utf-8") as f:
        data = json.load(f)
        for person in data:
            jobs = person['jobs']
            text = person['first_paragraph']
            to_many = False
            for job in jobs:
                if job in jobs_dict and not to_many:
                    if jobs_dict[job] < 500:
                        jobs_dict[job] = jobs_dict[job] + 1
                    else:
                        to_many = True
                else:
                    jobs_dict[job] = 1
            if not to_many:
                texts.append(text)
                persons.append(person['name'])
                labels.append(jobs)
"""
for jobs_in_label in labels:
    for job in jobs_in_label:
        if job in jobs:
            jobs[job] = jobs[job] + 1
        else:
            jobs[job] = 1
"""
# SORT LIST ON OCCURENCE
jobs_list = set(list(reversed(sorted(jobs_dict.items(), key=lambda item: item[1]))))
valid_labels = {}

# CREATE LABELS TO USE
for job in jobs_list:
    if job[1] > nr_labels_to_have:
        valid_labels[job[0]] = 1
        nr_labels = nr_labels + 1

for i, label in enumerate(labels):
    for j, l in enumerate(label):
        if l not in valid_labels:
            labels[i][j] = 'other'

other_count = 0
for i, label in enumerate(labels):
    for j, l in enumerate(label):
        if other_count > 2000 and labels[i][j] == 'other':
            del labels[i]
            del texts[i]
            del persons[i]
            break
        elif labels[i][j] == 'other':
            other_count = other_count + 1
            break

train = int(len(labels) * 0.8)
test = len(labels) - train

labels = labels[:(train+test)]
texts = texts[:(train+test)]

labels = one_hot.fit_transform(labels)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
persons = np.asarray(persons)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
persons = persons[indices]

x_train = data[:train]
y_train = labels[:train]
x_test = data[train:(train+test)]
y_test = labels[train:(train+test)]

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print(one_hot.classes_.shape)
print(y_test.shape)
pickle.dump(one_hot.classes_ , open('./good_stuff/valid_labels.p', 'wb'))

pickle.dump(x_train , open('./good_stuff/x_train.p', 'wb'))
pickle.dump(y_train , open('./good_stuff/y_train.p', 'wb'))
pickle.dump(x_test , open('./good_stuff/x_test.p', 'wb'))
pickle.dump(y_test , open('./good_stuff/y_test.p', 'wb'))
pickle.dump(persons , open('./good_stuff/persons.p', 'wb'))

## FOR WORD EMBEDDINGS
pickle.dump(word_index , open('./good_stuff/word_index.p', 'wb'))
pickle.dump(embedding_matrix , open('./good_stuff/embedding_matrix.p', 'wb'))
