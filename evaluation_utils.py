import numpy as np
import pprint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


def evaluate_architecture(model_path, dataset, problem_type="regression"):

    f = open(model_path + "parameters.txt")
    model_architecture = f.read()
    print(model_architecture)


def predict_hidden(model, hidden_dataset, feature_pp, problem_type="regression", target_pp=None):

    preds = model.forward(hidden_dataset[:, :3])
    preds = preds.detach().numpy()

    if problem_type == "classification":
        preds = preds.argmax(axis=1) # .squeeze()
        # One-hot encode the output - i.e. force the model to make a decision
        # TODO: Adam to get this working
        enc = LabelBinarizer()
        preds = enc.fit_transform(preds)
        
        pprint.pprint(preds)

    elif problem_type == "regression":

        # If this is a regression task and we normalised the output for training then revert it
        pprint.pprint(preds)
        if target_pp is not None:
            preds = target_pp.revert(preds)
            pprint.pprint(preds)
        else:
            pprint.pprint(preds)

    #features = feature_pp.revert(hidden_dataset[:, :3])
    #output = np.hstack((features, preds))
    #np.savetxt(problem_type + ".txt", output)
    return preds


def precision_calculator(confusion_matrix):
    """
    Calculates the precision rate in the confusion matrix
    i.e. for each class, number of correctly classified rows over total number of rows classified as that class
    :param confusion_matrix:
    :return:
    """
    precision_agg = 0
    for i in range(confusion_matrix.shape[0]):
        precision_agg += confusion_matrix[i, i] / confusion_matrix[:, i].sum()

    return precision_agg / confusion_matrix.shape[0]


def recall_calculator(confusion_matrix):
    """
    Calculates the recall rate in the confusion matrix
    :param confusion_matrix:
    :return:
    """
    recall_agg = 0
    for i in range(confusion_matrix.shape[0]):
        recall_agg += confusion_matrix[i, i] / confusion_matrix[i, :].sum()

    return recall_agg / confusion_matrix.shape[0]


def f1_score_calculator(precision, recall):
    """
    Calculates the f1 score as the "harmonic mean" of precision and recall
    :param precision:
    :param recall:
    :return:
    """
    return 2 * ((precision * recall) / (precision + recall))
