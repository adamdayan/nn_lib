import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from illustrate import illustrate_results_FM
from pytorch_pp import TorchPreprocessor


class FMDataset(Dataset):

    # Initialize your data, download, etc.
    def __init__(self, data):
        self.dataset = data
        self.len = self.dataset.shape[0]
        self.x_data = torch.from_numpy(self.dataset[:,:3])
        self.y_data = torch.from_numpy(self.dataset[:,3:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # #######################################################################
    # #                       ** END OF YOUR CODE **
    # #######################################################################
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
