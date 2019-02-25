import collections

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
 
def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    model = load_torch_model("test_save_model.pt", "test_save_layers.pickle")
    
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
