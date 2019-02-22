import collections

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM


class SequentialNet(nn.Module):

    def __init__(self, layers):

        super(SequentialNet, self).__init__()
        
        self._layer_names = layers
        self._layers = []

        # Chain together the layers
        for layer in self._layer_names:

            if layer.name == "linear":
                self._layers.append(nn.Linear(layer.in_dim, layer.out_dim))
            elif layer.name == "relu":
                self._layers.append(nn.ReLU())
            elif layer.name == "sigmoid":
                self._layers.append(nn.Sigmoid())
            elif layer.name == "dropout":
                self._layers.append(nn.Dropout(p=layer.p))

        self.net = nn.Sequential(*self._layers)

    def forward(self, x):
        return self.net(x)

class Trainer():

    def __init__(self,
                 network,
                 batch_size,
                 nb_epoch,
                 learning_rate,
                 loss_fun,
                 shuffle_flag):

        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

    @staticmethod
    def shuffle():
        pass

    def train(self, input_dataset, target_dataset):
        pass

    def eval_loss(self, input_dataset, target_dataset):
        pass

    


def split_train_val_test(dataset, last_feature_idx):

    np.random.shuffle(dataset)
    x = dataset[:, :(last_feature_idx + 1)]
    y = dataset[:, (last_feature_idx + 1):]
    
    # TODO: CHECK THE SPLIT THOROUGLY
    # Split the dataset into train, val, test
    train_idx = int(0.8 * len(x))

    x_train = x[:train_idx]
    y_train = y[:train_idx]

    # Remainder should be split 
    x_rem = x[train_idx:]
    y_rem = y[train_idx:]

    val_idx = int(0.5 * len(x_rem))

    x_val = x_rem[:val_idx]
    y_val = y_rem[:val_idx]

    x_test = x_rem[val_idx:]
    y_test = x_rem[val_idx:]

    print("Input data split into train, val, test with shapes:")
    print("- x_train = " + str(x_train.shape))
    print("- y_train = " + str(y_train.shape))
    print("- x_val = " + str(x_val.shape))
    print("- y_val = " + str(y_val.shape))
    print("- x_test = " + str(x_test.shape))
    print("- y_test = " + str(y_test.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test

def train():

    # Load and prepare data 
    dataset = np.loadtxt("FM_dataset.dat")

    # Split data
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(dataset, 2)

    # TODO: preprocess the data 
    
    # Instatiate a network
    LinearLayer = collections.namedtuple("LinearLayer", "name in_dim out_dim")
    ReluLayer = collections.namedtuple("ReluLayer", "name")
    ReluLayer = collections.namedtuple("SigmoidLayer", "name")
    DropoutLayer = collections.namedtuple("DropoutLayer", "name p")
    
    layers = [LinearLayer(name="linear", in_dim=3, out_dim=8),
              LinearLayer(name="linear", in_dim=8, out_dim=8),
              ReluLayer(name="relu"),
              DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=8, out_dim=8),
              ReluLayer(name="relu"),
              DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=3, out_dim=8)]

    network = SequentialNet(layers)
    print("Network instatiated:")
    print(network)

    # Add the network to a trainer and train


    # Evaluate results 
    
    # out = network.forward(dataset
    

    

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_FM(network, prep)


if __name__ == "__main__":
    train()
    # main()



# class Net(nn.Module):
#     def __init__(self, input_dim, neurons, activations):

#         super(Net, self).__init__()

#         self.input_dim = input_dim
#         self.neurons = neurons
#         self.activations = activations
        
#         self._layers = []

#         # Chain together the layers
#         for idx in range(len(neurons)):

#             if idx == 0:
#                 self._layers.append(nn.Linear(self.input_dim, self.neurons[0]))
#             else:
#                 self._layers.append(nn.Linear(self.neurons[idx-1], self.neurons[idx]))

#     def forward(self, x):

#         return = self.net(x)


#         # Iterate through layers in layer to perform forward pass
#         for idx, layer in enumerate(self._layers):

#             # If it's the input layer, there is no activation function
#             if idx == 0:
#                 x = self._layers[idx](x)

#             # Else, there is an activation function 
#             else:
#                 if self.activations[idx] == "relu":
#                     x = self.chain_relu(_layers[idx]())
#                 elif self.activation[idx] == "sigmoid":
#                     pass
#                 else:
#                     x = self._layers[idx](x)
            

#     def chain_relu(self, x):
#         return F.relu(x)

#     def chain_sigmoid(self, x):
#         pass
