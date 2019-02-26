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



def evaluate_architecture(model_path,dataset,problem_type):
    #w = torch.load(model_path)
    checkpoint = torch.load(model_path+"20190226_203050_model_model.pt")

    model = load_torch_model(model_path + "20190226_203050_model_model.pt", model_path + "20190226_203050_model_layers.pickle")
    f=open(model_path+"parameters.txt")
    model_hyperparams = f.read()
    print(model_hyperparams)
    if problem_type== "classification":
        train_preds= model.forward(dataset[:, :3])
        train_preds = train_preds.detach().numpy().argmax(axis=1).squeeze()
        train_targets = (dataset[:, 3:6]).argmax(axis=1).squeeze()
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





    #loss=(checkpoint['Loss'])



def main():

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    dataset = np.loadtxt("FM_dataset.dat")
    model_path = "output/learn_fm/20190226_203050/"
    problem_type="classification"
    evaluate_architecture(model_path,dataset,problem_type)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()





    # # Plot learning curves
    # # to check how well model is training (e.g. is there overfitting)
    # plt.figure(figsize=(20,10))
    # plt.suptitle("Loss and accuracy vs epochs for " + loss_fun)

    # # Description of the hyperparams
    # hyperparams_text = "Hyperparameters: \n " + \
    #                    "- lr = " + str(learning_rate) + \
    #                    "\n - batch_size = " + str(batch_size) + \
    #                    "\n - loss function = " + loss_fun + \
    #                    "\n - number of epochs = " + str(nb_epoch)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.figtext(0.75, 0.75, hyperparams_text, bbox=props)

    # # Plot loss vs number of epochs
    # plt.plot(trainer.epochs_w_loss_measure, trainer.training_losses)
    # plt.plot(trainer.epochs_w_loss_measure, trainer.validation_losses)
    # plt.legend(['training', 'validation'], loc='upper left')
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Loss (" + loss_fun + ")")

    # plt.savefig(loss_fun + "_loss_plot.png")

    # out = network.forward(dataset
