import numpy as np


def precision_calculator(confusion_matrix):
    """
    Calculates the precision rate in the confusion matrix
    i.e. for each class, number of correctly classified rows over total number of rows classified as that class
    :param confusion_matrix:
    :return:
    """
    precision_agg = 0
    for i in range(confusion_matrix.shape[0]):
        precision_agg += confusion_matrix[i, i] / confusion_matrix[i, :].sum()

    return precision_agg / confusion_matrix.shape[0]


def recall_calculator(confusion_matrix):
    """
    Calculates the recall rate in the confusion matrix
    :param confusion_matrix:
    :return:
    """
    recall_agg = 0
    for i in range(confusion_matrix.shape[0]):
        recall_agg += confusion_matrix[i, i] / confusion_matrix[:, i].sum()

    return recall_agg / confusion_matrix.shape[0]


def classification_rate_calculator(confusion_matrix):
    """
    Calculates the classification rate in the confusion matrix
    :param confusion_matrix:
    :return:
    """
    diagonal_condition = np.eye(confusion_matrix.shape[0], dtype=bool)
    return confusion_matrix[diagonal_condition].sum() / confusion_matrix.sum()


def f1_score_calculator(precision, recall):
    """
    Calculates the f1 score as the "harmonic mean" of precision and recall
    :param precision:
    :param recall:
    :return:
    """
    return 2 * ((precision * recall) / (precision + recall))
