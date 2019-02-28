import numpy as np
import pprint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder


def evaluate_architecture(model_path, dataset, problem_type="regression"):

    f = open(model_path + "parameters.txt")
    model_architecture = f.read()
    print(model_architecture)

    # if problem_type== "classification":
        
    #     train_preds = model.forward(dataset[:, :3])
    #     train_preds = train_preds.detach().numpy().argmax(axis=1).squeeze()
    #     train_targets = (dataset[:, 3:6]).argmax(axis=1).squeeze()

    #     print("Classification Confusion Matrix:")
    #     conf_matrix = confusion_matrix(train_targets, train_preds)
    #     print(conf_matrix)
        
    #     recall = recall_calculator(conf_matrix)
    #     print("Recall: ", recall)

    #     precision = precision_calculator(conf_matrix)
    #     print("Precision: ", precision)

    #     f1 = f1_score_calculator(precision,recall)
    #     print("F1 Score: ", f1)


def predict_hidden(model, hidden_dataset, problem_type="regression"):

    preds = model.forward(hidden_dataset[:, :3])
    preds = preds.detach().numpy()
    pprint.pprint(preds)

    if problem_type == "classification":
        preds = preds.argmax(axis=1) # .squeeze()
        # One-hot encode the output - i.e. force the model to make a decision
        # TODO: Adam to get this working
        enc = OneHotEncoder(handle_unknown="ignore")
        preds = enc.fit_transform(preds).toarray()
        pprint.pprint(preds)


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
