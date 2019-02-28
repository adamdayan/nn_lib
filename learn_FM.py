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

    # out = network.forward(dataset
