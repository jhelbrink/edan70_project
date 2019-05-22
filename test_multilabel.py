




import os
import json
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Input, LSTM, Dropout, Activation, Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


max_num_words = 5000
max_seq_length = 250

embedding_dim = 100

labelencoder = LabelEncoder()

glove_dir = './glove.6B'

embedding_dim = 100
maxlen = 250
#926796 complete
#training_samples = 100000
train = 100000

#231700 complete
validation_samples = 20000
test = 20000
max_words = 10000


texts = []
labels = []

## EXTRACT WORDS AND COUNT HOW OFTEN THEY ARE USED
for filename in os.listdir('person_data2'):
    with open('person_data2/' + filename, encoding="utf-8") as f:
        data = json.load(f)
        for person in data:
            jobs = person['jobs']
            text = person['first_paragraph']
            for job in jobs:
                texts.append(text)
                labels.append(job)

tokenizer = Tokenizer(num_words=max_num_words, lower=True)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print("Found %s unique tokens" %len(word_index))

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen = max_seq_length)
print("Shape of text tensor: ", X.shape)

labels = pd.get_dummies(labels)
Y = labels
print("Shape of label tensor: ", Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(max_num_words, embedding_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4638, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 5
batch_size = 100

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])



"""





texts = []
labels = []

cur_label = 1

label_translate = {}


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
"""