import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense, Input, LSTM, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

x_train = pickle.load(open('./good_stuff/x_train.p', 'rb'))
y_train = pickle.load(open('./good_stuff/y_train.p', 'rb'))
#x_test = pickle.load(open('./good_stuff/x_val.p','rb'))
#y_test = pickle.load(open('./good_stuff/y_val.p','rb'))
embedding_matrix = pickle.load(open('./good_stuff/embedding_matrix.p', 'rb'))

#ENCODE
y_train = to_categorical(y_train)
print(y_train.shape)

embedding_dim = 100
maxlen = 60
max_words = 10000

model = Sequential()
model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix]))
model.add(LSTM(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(512, input_shape=(max_words,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4632, activation='sigmoid'))    
#model.layers[0].set_weights([embedding_matrix])
#model.layers[0].trainable = False

    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,batch_size=100, verbose=1, validation_split=0.1, epochs=5)

#model.save_weights('./pre_trained_glove_model.h5')

"""
inp = Input(shape=(maxlen,))
x = Embedding(max_words, embedding_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
x = GlobalMaxPool1D()(x)
x = Dense(60, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(4627, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.1, verbose=True);
"""
loss,accuracy = model.evaluate(x_train,y_train)
print('Training accuracy: {:.4f}'.format(accuracy))
      
#model.save_weights('./pre_trained_glove_model.h5')

# import matplotlib.pyplot as plt
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
    
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()