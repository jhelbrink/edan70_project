from keras.models import load_model
from keras.models import model_from_json
import json
import pickle
from plot_cf_matrix import plot_confusion_matrix
from evaluate_f1 import evaluate_f1
from most_frequent_labels import most_frequent_indices
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
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

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

x_test = pickle.load(open('./good_stuff/x_test.p', 'rb'))
y_test = pickle.load(open('./good_stuff/y_test.p', 'rb'))
labels = pickle.load(open('./good_stuff/valid_labels.p', 'rb'))
persons = pickle.load(open('./good_stuff/persons.p', 'rb'))

prediction = model.predict(x_test)


print(persons[80014])

for i, val in enumerate(prediction[14]):
    if val > 0.05:
        print(val)
        print(labels[i])
print(' ')
for i, val in enumerate(y_test[14]):
    if val > 0.5:
        print(val)
        print(labels[i])

# Compute best threshhold
#best_score = evaluate_f1(y_test, prediction)
prediction = prediction > 0.18

frequency_index = {}

most_frequent_filled = []

for y in y_test:
    for i, v in enumerate(y):
        if v == 1:
            if i in frequency_index:
                frequency_index[i] = frequency_index[i] + 1
            else:
                frequency_index[i] = 1

most_frequent = list(reversed(sorted(frequency_index.items(), key=lambda item: item[1])))[:25]

for index in most_frequent:
    most_frequent_filled.append(index[0])

most_frequent_indices = most_frequent_filled

matrix_y_test = break_out_most_frequent_labels_in_y(y_test, most_frequent_indices)
matrix_y_pred = break_out_most_frequent_labels_in_y(prediction, most_frequent_indices)
matrix_labels = np.asarray([labels[i] for i in most_frequent_indices])

matrix = confusion_matrix(matrix_y_test.argmax(axis=1), matrix_y_pred.argmax(axis=1))

print(matrix.shape)

plt.figure()
plot_confusion_matrix(matrix, classes=matrix_labels,
                      title='Confusion matrix, without normalization')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
"""
print(matrix)
df_cm = pd.DataFrame(matrix, index = [i for i in labels[:500]],
                  columns = [i for i in labels[:500]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
"""
"""
conf_mat_dict={}

for label_col in range(len(labels)):
    y_true_label = y_true[:, label_col]
    y_pred_label = y_pred[:, label_col]
    conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


for label, matrix in conf_mat_dict.items():
    print("Confusion matrix for label {}:".format(label))
    print(matrix)
"""
