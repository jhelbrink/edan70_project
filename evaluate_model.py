from keras.models import load_model
from keras.models import model_from_json
import json
import pickle
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

matrix = confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))

plt.figure()
plot_confusion_matrix(matrix, classes=labels,
                      title='Confusion matrix, without normalization')
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
