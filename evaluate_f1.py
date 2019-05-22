from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def evaluate_f1(y_test, prediction):
    prediction_copy = prediction
    best_f1_score = 0
    optimal_precision = 0
    optimal_recall = 0
    best_threshhold = 0.1
    best_precision = 0
    best_threshhold_precision = 0.1
    best_recall = 0
    best_threshhold_recall = 0.1
    for threshhold in np.linspace(0.05,0.99,95):
        prediction = (prediction_copy > threshhold)
        score = f1_score(y_test, prediction, average='micro')
        precision = precision_score(y_test, prediction, average='micro')
        recall = recall_score(y_test, prediction, average='micro')
        if score > best_f1_score:
            best_f1_score = score
            best_threshhold = threshhold
            optimal_precision = precision
            optimal_recall = recall
        if precision > best_precision:
            best_precision = precision
            best_threshhold_precision = threshhold
        if recall > best_recall:
            best_recall = recall
            best_threshhold_recall = threshhold

    print('Best thresh', best_threshhold)
    print('Best score', best_f1_score)
    print('Optimal precision', optimal_precision)
    print('Optimal recall', optimal_recall)
    print('Best recall', best_recall)
    print('Best precision', best_precision)
    print('Best threshold precision', best_threshhold_precision)
    print('Best threshold recall', best_threshhold_recall)
    return best_threshhold
