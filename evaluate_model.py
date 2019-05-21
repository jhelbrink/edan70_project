from keras.models import load_model
from keras.models import model_from_json
import json
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



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

print(y_test.shape)
print(prediction.shape)
prediction = (prediction > 0.2)
y_test = (y_test > 0.5)

matrix = confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))
print(matrix)
df_cm = pd.DataFrame(matrix, index = [i for i in labels[:500]],
                  columns = [i for i in labels[:500]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

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
