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

            print(self._layers)
            if layer.name == "linear":
                self._layers.append(nn.Linear(layer.in_dim, layer.out_dim))
            elif layer.name == "relu":
                self._layers.append(nn.ReLU())
            elif layer.name == "dropout":
                self._layers.append(nn.Dropout(p=layer.p))

        self.net = nn.Sequential(*self._layers)

    def forward(self, x):
        return self.net(x)

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
    

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    LinearLayer = collections.namedtuple("LinearLayer", "name in_dim out_dim")
    ReluLayer = collections.namedtuple("ReluLayer", "name")
    DropoutLayer = collections.namedtuple("DropoutLayer", "name p")
    
    layers = [LinearLayer(name="linear", in_dim=3, out_dim=8),
              LinearLayer(name="linear", in_dim=8, out_dim=8),
              ReluLayer(name="relu"),
              DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=8, out_dim=8),
              ReluLayer(name="relu"),
              DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=3, out_dim=8)]

    sequential_net = SequentialNet(layers)

    print(sequential_net)
    
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
