import os
import json
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

glove_dir = './glove.6B'

embedding_dim = 100
maxlen = 80
training_samples = 926796
validation_samples = 231700
max_words = 6500

texts = []
labels = []

cur_label = 1

label_translate = {}

## EXTRACT WORDS AND COUNT HOW OFTEN THEY ARE USED
for filename in os.listdir('person_data2'):
    with open('person_data2/' + filename) as f:
        data = json.load(f)
        for person in data:
            jobs = person['jobs']
            text = person['first_paragraph']
            for job in jobs:
                texts.append(text)
                labels.append(job)

#labels = pd.get_dummies(labels)
labels = labelencoder.fit_transform(labels)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

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

#x = preprocessing.sequence.pad_sequences(x, maxlen=maxlen)
pickle.dump(x_train , open('./good_stuff/x_train.p', 'wb'))
pickle.dump(y_train , open('./good_stuff/y_train.p', 'wb'))
pickle.dump(x_val , open('./good_stuff/x_val.p', 'wb'))
pickle.dump(y_val , open('./good_stuff/y_val.p', 'wb'))
pickle.dump(word_index , open('./good_stuff/word_index.p', 'wb'))
pickle.dump(embedding_matrix , open('./good_stuff/embedding_matrix.p', 'wb'))
