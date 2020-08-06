"""
This module contains the different metrics to evaluate the quality of predictions
sklearn anyway implements all of them
Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> pred = [int(np.round(i)) for i in np.random.rand(20)]
    >>> Y = [int(np.round(i)) for i in np.random.rand(20)]
    >>> _get_number_of_true_positive(pred, Y)
    11
    >>> _get_number_of_true_negative(pred, Y)
    4
    >>> _get_number_of_false_negative(pred, Y)
    2
    >>> _get_number_of_false_positive(pred, Y)
    3
    >>> get_recall(pred, Y)
    0.8461538461538461
    >>> get_accuracy(pred, Y)
    0.75
    >>> get_precision(pred, Y)
    0.7857142857142857
    >>> get_f1_score(pred, Y)
    0.8148148148148148
"""
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


def _get_diagonal_elements(mat):
    return [mat[i][i] for i in range(len(mat))]


def get_accuracy(pred, y): # also np.sum(_get_diagonal_elements(confusion_matrix(pred, Y))) / len(Y)
    return (pred == y).mean()


def _get_f1_from_precision_and_recall(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def get_f1_score(pred, Y):
    return _get_f1_from_precision_and_recall(get_precision(pred, Y), get_recall(pred, Y))


def get_recall(pred, Y):
    """    Returns recall defined as: TP / (FN + TP) (true positive / condition positive)    """
    TP = _get_number_of_true_positive(pred, Y)
    FN = _get_number_of_false_negative(pred, Y)
    return TP / (TP + FN)


def get_precision(pred, Y):
    """    Returns precision defined as: TP / (TP + FP)    """
    TP = _get_number_of_true_positive(pred, Y)
    FP = _get_number_of_false_positive(pred, Y)
    return TP / (TP + FP)


# True/False - Positive/Negative (All of these can be also done with sklearn.metrics.confusion_matrix)
def _count_values(pred, y, pred_val, y_val):
    return np.sum(np.logical_and(np.array(pred) == pred_val, np.array(y) == y_val))

def _get_number_of_true_positive(pred, Y): return _count_values(pred, Y, 1, 1)
def _get_number_of_true_negative(pred, Y): return _count_values(pred, Y, 0, 0)
def _get_number_of_false_negative(pred, Y): return _count_values(pred, Y, 0, 1)
def _get_number_of_false_positive(pred, Y): return _count_values(pred, Y, 1, 0)


def get_confusion_matrix_df(pred, Y):
    return pd.DataFrame(data=confusion_matrix(y_pred=pred, y_true=Y),
                        columns=['true_0', 'true_1'],
                        index=['pred_0', 'pred_1'],
                        )