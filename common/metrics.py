"""
This module contains the different metrics to evaluate the quality of predictions
sklearn anyway implements all of them
Examples:
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> pred = [int(np.round(i)) for i in np.random.rand(20)]
    >>> y = [int(np.round(i)) for i in np.random.rand(20)]
    >>> _get_number_of_true_positive(pred, y)
    11
    >>> _get_number_of_true_negative(pred, y)
    4
    >>> _get_number_of_false_negative(pred, y)
    2
    >>> _get_number_of_false_positive(pred, y)
    3
    >>> get_recall(pred, y)
    0.8461538461538461
    >>> get_accuracy(pred, y)
    0.75
    >>> get_precision(pred, y)
    0.7857142857142857
    >>> get_f1_score(pred, y)
    0.8148148148148148
"""
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


def _get_diagonal_elements(mat):
    return [mat[i][i] for i in range(len(mat))]


def get_accuracy(pred: np.array, y: np.array):  # or np.sum(_get_diagonal_elements(confusion_matrix(pred, y))) / len(y)
    return (pred == y).mean()


def _get_f1_from_precision_and_recall(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def get_f1_score(pred, y):
    return _get_f1_from_precision_and_recall(get_precision(pred, y), get_recall(pred, y))


def get_recall(pred, y):
    """    Returns recall defined as: tp / (fn + tp) (true positive / condition positive)    """
    tp = _get_number_of_true_positive(pred, y)
    fn = _get_number_of_false_negative(pred, y)
    return tp / (tp + fn)


def get_precision(pred, y):
    """    Returns precision defined as: tp / (tp + fp)    """
    tp = _get_number_of_true_positive(pred, y)
    fp = _get_number_of_false_positive(pred, y)
    return tp / (tp + fp)


# True/False - Positive/Negative (All of these can be also done with sklearn.metrics.confusion_matrix)
def _count_values(pred, y, pred_val, y_val):
    return np.sum(np.logical_and(np.array(pred) == pred_val, np.array(y) == y_val))


def _get_number_of_true_positive(pred, y): return _count_values(pred, y, 1, 1)
def _get_number_of_true_negative(pred, y): return _count_values(pred, y, 0, 0)
def _get_number_of_false_negative(pred, y): return _count_values(pred, y, 0, 1)
def _get_number_of_false_positive(pred, y): return _count_values(pred, y, 1, 0)


def get_confusion_matrix_df(pred, y):
    return pd.DataFrame(data=confusion_matrix(y_pred=pred, y_true=y),
                        columns=['true_0', 'true_1'],
                        index=['pred_0', 'pred_1'],
                        )
