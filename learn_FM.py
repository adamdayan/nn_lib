import collections

from tqdm import tqdm
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

class TorchTrainer():

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

        # Clear training loss metadata from any previous training runs
        self.epochs_w_loss_measure = []
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        assert input_dataset.shape[0] == target_dataset.shape[0], "input and target dataset do not have same number of datapoints!"
        shuffled_idxs = np.random.permutation(input_dataset.shape[0])
        return input_dataset[shuffled_idxs], target_dataset[shuffled_idxs]

        

    def minibatch_data(self, input_dataset, target_dataset):
        assert input_dataset.shape[0] == target_dataset.shape[0], "input and target dataset do not have same number of datapoints!"

        cut_points = np.arange(self.batch_size, input_dataset.shape[0], self.batch_size)
        return np.split(input_dataset, cut_points), np.split(target_dataset, cut_points)

    def train(self, input_dataset, target_dataset):

        # Clear training loss metadata from any previous training runs
        self.epochs_w_loss_measure = []
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

        # Shuffle input data
        if self.shuffle_flag:
            print("Shuffling dataset")
            input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)

        # Split datasets into batches
        batched_input, batched_target = self.minibatch_data(input_dataset, target_dataset)
        print("Number of batches = ", len(batched_input))
        print("Batch size = ", batched_input[0].shape)
        
        # Start outer training loop for epoch
        for epoch in tqdm(range(1, self.nb_epoch + 1)):
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
    x_train_pre = x_train
    x_val_pre = x_val
    x_test_pre = x_test
    
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
    # TODO: add to a tuple or something more secure (link it with the hyperparameter optimiser)
    batch_size = 8
    nb_epoch = 1000
    learning_rate = 0.01
    loss_fun = "mse"
    shuffle_flag = True
    
    trainer = TorchTrainer(
        network=network,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        learning_rate=learning_rate,
        loss_fun=loss_fun,
        shuffle_flag=shuffle_flag,
    )

    trainer.train(x_train_pre, y_train)


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
