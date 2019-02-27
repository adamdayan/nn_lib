import collections
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from illustrate import illustrate_results_FM
from pytorch_pp import TorchPreprocessor

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
from evaluation_utils import *

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



def evaluate_architecture(model_path,dataset):
    #w = torch.load(model_path)
    #checkpoint = torch.load(model_path+"20190226_203050_model_model.pt")
    f=open(model_path+"parameters.txt")
    model_architecture = f.read()
    print(model_architecture)

    #loss=(checkpoint['Loss'])

def predict_hidden(model,hidden_dataset):
    train_preds= model.forward(hidden_dataset[:, :3])
    print(train_preds)
    train_preds = train_preds.detach().numpy().argmax(axis=1).squeeze()
    print(train_preds)


>>>>>>> evaluation of trained network

def main():

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    dataset = np.loadtxt("FM_dataset.dat")
    model_path = "output/learn_fm/20190227_123832/"
    model = load_torch_model(model_path + "20190227_123832_model_model.pt", model_path + "20190227_123832_model_layers.pickle")
    evaluate_architecture(model_path, dataset)
    predict_hidden(model,dataset)

<<<<<<< HEAD
    # #######################################################################
    # #                       ** END OF YOUR CODE **
    # #######################################################################
=======

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
>>>>>>> evaluation of trained network
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
