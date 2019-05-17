import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense, Input, LSTM, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.utils.np_utils import to_categorical
# import matplotlib.pyplot as plt

# plt.style.use('ggplot')

# def plot_history(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
    
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()


x_train = pickle.load(open('./good_stuff/x_train.p', 'rb'))
y_train = pickle.load(open('./good_stuff/y_train.p', 'rb'))
embedding_matrix = pickle.load(open('./good_stuff/embedding_matrix.p', 'rb'))

#ENCODE
y_train = to_categorical(y_train)
print(y_train.shape)

embedding_dim = 100
maxlen = 80
max_words = 15000


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

inp = Input(shape=(maxlen,))
x = Embedding(max_words, embedding_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
x = GlobalMaxPool1D()(x)
x = Dense(60, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(4638, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.1);

#loss,accuracy = model.evaluate(x_train,y_train, verbose = false)
#print('Training accuracy: {:.4f}'.format(accuracy))
      
model.save_weights('./pre_trained_glove_model.h5')

plot_history(history)
      

