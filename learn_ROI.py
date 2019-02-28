import collections
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch_utils import *

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
from evaluation_utils import *


def evaluate_architecture(model_path,dataset):
    #w = torch.load(model_path)
    checkpoint = torch.load(model_path+"20190227_112136_model_model.pt")

    model = load_torch_model(model_path + "20190227_112136_model_model.pt", model_path + "20190227_112136_model_layers.pickle")
    f=open(model_path+"parameters.txt")
    model_hyperparams = f.read()
    print(model_hyperparams)


    train_preds= model.forward(dataset[:, :3])
    train_preds = train_preds.detach().numpy().argmax(axis=1).squeeze()
    train_targets = (dataset[:, 3:6]).argmax(axis=1).squeeze()
    print(train_targets)
    print(train_preds)
    print("Classification Confusion Matrix")
    conf_matrix=confusion_matrix(train_targets, train_preds)
    print(conf_matrix)
    print("\n")
    recall=recall_calculator(conf_matrix)
    print("Recall: ", recall)
    precision=precision_calculator(conf_matrix)
    print("Precision: ", precision)
    f1=f1_score_calculator(precision,recall)
    print("F1 Score: ", f1)


def main():

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    dataset = np.loadtxt("ROI_dataset.dat")
    model_path = "output/learn_roi/20190227_112136/"

    evaluate_architecture(model_path,dataset)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_ROI(network, prep)


if __name__ == "__main__":
    main()
