from keras.models import load_model
from keras.models import model_from_json
import json
import pickle
from functions.plot_cf_matrix import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def break_out_most_frequent_labels_in_y(y, indices):
    new_array = []
    for val_array in y:
        temp_array = []
        for i, val in enumerate(val_array):
            if i in indices:
                temp_array.append(val)
        new_array.append(temp_array)
    return np.asarray(new_array)

dir_to_load_from = ''
model_to_load = ''
model_weights_to_load = ''

print('What model do you want to evaluate?')
print('1. 50-500')
print('2. 100-500')
model_to_eval = input()

if dataset == '1':
    print('Evaluating the larger model')
    min_jobs = 50
    dir_to_load_from = './50-500-data/'
    model_to_load = './models/model_in_json_big.json'
    model_weights_to_load = './models/model_weights_big.h5'
elif dataset == '2':
    print('Evaluating the smaller model')
    min_jobs = 100
    dir_to_load_from = './100-500-data/'
    model_to_load = './models/model_in_json_big.json'
    model_weights_to_load = './models/model_weights_big.h5'
else:
    sys.exit()

with open(model_to_load,'r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(model_weights_to_load)

x_test = pickle.load(open(dir_to_load_from + 'x_test.p', 'rb'))
y_test = pickle.load(open(dir_to_load_from + 'y_test.p', 'rb'))
labels = pickle.load(open(dir_to_load_from + 'valid_labels.p', 'rb'))
persons = pickle.load(open(dir_to_load_from + 'persons.p', 'rb'))
y_train = pickle.load(open(dir_to_load_from + 'y_train.p', 'rb'))
translate_label_dict = pickle.load(open(dir_to_load_from + 'translate_label_dict.p', 'rb'))

cut = len(y_train)

persons = persons[cut:] #FORMAT PERSON TO FIT TEST DATA

prediction = model.predict(x_test)

# Compute best threshhold
prediction = prediction > 0.22
best_score = evaluate_f1(y_test, prediction)
score = f1_score(y_test, prediction, average='micro')
precision = precision_score(y_test, prediction, average='micro')
recall = recall_score(y_test, prediction, average='micro')
print('F1-SCORE', score)
print('Precision', precision)
print('RECALL', recall)

print('Plotting confusion matrix...')

#USE TO FIND MOST FREQUENT LABELS
frequency_index = {}

most_frequent_filled = []

for y in y_test:
    for i, v in enumerate(y):
        if v == 1:
            if i in frequency_index:
                frequency_index[i] = frequency_index[i] + 1
            else:
                frequency_index[i] = 1

#25 most frequent labels
most_frequent = list(reversed(sorted(frequency_index.items(), key=lambda item: item[1])))[:25]

for index in most_frequent:
    most_frequent_filled.append(index[0])

most_frequent_indices = most_frequent_filled

matrix_y_test = break_out_most_frequent_labels_in_y(y_test, most_frequent_indices)
matrix_y_pred = break_out_most_frequent_labels_in_y(prediction, most_frequent_indices)
matrix_labels = np.asarray([labels[i] for i in most_frequent_indices])

for i, l in enumerate(matrix_labels):
    if l != 'other':
        matrix_labels[i] = translate_label_dict[l]

matrix = confusion_matrix(matrix_y_test.argmax(axis=1), matrix_y_pred.argmax(axis=1))

plt.figure()
plot_confusion_matrix(matrix, classes=matrix_labels,
                      title='Confusion matrix, without normalization')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
