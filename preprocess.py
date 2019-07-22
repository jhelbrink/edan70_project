import os
import json
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import sys

one_hot = MultiLabelBinarizer()

dir_to_save_in = ''
min_jobs = 50 #NUMBER OF OCCURENCES OF OCCUPATION TO INCLUDE
max_jobs = 500 #MAXIMUM NUMBER OF LABELS IN SET
max_other = 1500 #MAXIMUM OTHER OCCURENCES

glove_dir = './'

embedding_dim = 100
maxlen = 150
max_words = 10000

texts = []
labels = []
persons = []

print('What limit do you want on your occupations?')
print('1. 50-500')
print('2. 100-500')
dataset = input()

if dataset == '1':
    print('Processing the larger model')
    min_jobs = 50
    dir_to_save_in = './50-500-data/'
elif dataset == '2':
    print('Processing the smaller model')
    min_jobs = 100
    dir_to_save_in = './100-500-data/'
else:
    sys.exit()

# Use to keep score of how many occurences of a job
jobs_dict = {}

## EXTRACT WORDS AND COUNT HOW OFTEN THEY ARE USED
for filename in os.listdir('person_data'):
    with open('person_data/' + filename, encoding="utf-8") as f:
        data = json.load(f)
        for person in data:
            jobs = person['jobs']
            text = person['first_paragraph']
            to_many = False
            for job in jobs:
                if job in jobs_dict and not to_many:
                    if jobs_dict[job] < max_jobs:
                        jobs_dict[job] = jobs_dict[job] + 1
                    else:
                        to_many = True
                else:
                    jobs_dict[job] = 1
            if not to_many:
                texts.append(text)
                persons.append(person['name'])
                labels.append(jobs)

# SORT LIST ON OCCURENCE
jobs_list = set(list(reversed(sorted(jobs_dict.items(), key=lambda item: item[1]))))
valid_labels = {}

# CREATE LABELS TO USE
for job in jobs_list:
    if job[1] > min_jobs:
        valid_labels[job[0]] = 1 # SETS IF INCLUDE JOB

# SET LABELS TO OTHER THAT ARE NO IN THE THRESHOLD
for i, label in enumerate(labels):
    for j, l in enumerate(label):
        if l not in valid_labels:
            labels[i][j] = 'other'

# DELETE OTHER OCCURENCES ABOVE LIMIT, THIS CAN DESTROY THE SET A LITTLE BIT BUT WE WILL SEE PAST THAT FOR NOW
other_count = 0
for i, label in enumerate(labels):
    for j, l in enumerate(label):
        if other_count > max_other and labels[i][j] == 'other':
            del labels[i]
            del texts[i]
            del persons[i]
            break
        elif labels[i][j] == 'other':
            other_count = other_count + 1
            break

train = len(texts) * 0.8 #TRAIN SIZE
test = len(texts) - train #TEST SIZE

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

pickle.dump(one_hot.classes_ , open(dir_to_save_in + 'valid_labels.p', 'wb'))

pickle.dump(x_train , open(dir_to_save_in + 'x_train.p', 'wb'))
pickle.dump(y_train , open(dir_to_save_in + 'y_train.p', 'wb'))
pickle.dump(x_test , open(dir_to_save_in + 'x_test.p', 'wb'))
pickle.dump(y_test , open(dir_to_save_in + 'y_test.p', 'wb'))
pickle.dump(persons , open(dir_to_save_in + 'persons.p', 'wb'))

## FOR WORD EMBEDDINGS
pickle.dump(word_index , open(dir_to_save_in + 'word_index.p', 'wb'))
pickle.dump(embedding_matrix , open(dir_to_save_in + 'embedding_matrix.p', 'wb'))
