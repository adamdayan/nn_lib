import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SequentialNet(nn.Module):

    def __init__(self, layers, device="cpu"):

        super(SequentialNet, self).__init__()

        self._layer_names = layers
        self._layers = []

        self.device = device

        # Chain together the layers
        for layer in self._layer_names:

            if layer.name == "linear":
                self._layers.append(nn.Linear(layer.in_dim, layer.out_dim))
            elif layer.name == "relu":
                self._layers.append(nn.ReLU())
            elif layer.name == "sigmoid":
                self._layers.append(nn.Sigmoid())
            elif layer.name == "tanh":
                self._layers.append(nn.Tanh())
            elif layer.name == "dropout":
                self._layers.append(nn.Dropout(p=layer.p))
            elif layer.name == "softmax":
                self._layers.append(nn.Softmax())

        self.network = nn.Sequential(*self._layers)

    def forward(self, x):
        x_tens = torch.from_numpy(x).float()
        x_tens = x_tens.to(self.device)
        return self.network(x_tens)