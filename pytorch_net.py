import collections
import time
import datetime
import os
import sys
import getopt

from logger import Logger
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pp import TorchPreprocessor
from sklearn.metrics import confusion_matrix
from torch_utils import *
from evaluation_utils import *

from imblearn.over_sampling import SMOTE, RandomOverSampler

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


class TorchTrainer():

    def __init__(self,
                 problem_type,
                 network,
                 batch_size,
                 nb_epoch,
                 learning_rate,
                 loss_fun,
                 shuffle_flag,
                 optimizer,
                 device,
                 log_output_path=''):

        self.problem_type = problem_type
        self.network = network.to(device)
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag
        self.device = device
        self.log_output_path = log_output_path + '/logs'
        self.logger = Logger(self.log_output_path)

        if self.loss_fun == "mse":
            self.loss_criterion = nn.MSELoss()
        elif self.loss_fun == "cross_entropy":
            self.loss_criterion = nn.CrossEntropyLoss()

        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

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

    def train(self, input_dataset, target_dataset, input_dataset_val, target_dataset_val):

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
        for epoch in range(1, self.nb_epoch + 1):

            self.network.train()  # Sets the model back to training mode

            # Start inner training loop forwards and backwards pass for each batch
            for idx, batch_input in enumerate(batched_input):

                # Forward pass through the network
                batch_output = self.network.forward(batch_input)

                # Compute loss
                if self.problem_type == "regression":
                    batch_target = torch.from_numpy(batched_target[idx]).float()
                elif self.problem_type == "classification":
                    # https://discuss.pytorch.org/t/loss-functions-for-batches/20488
                    batch_target = torch.from_numpy(batched_target[idx]).type(torch.long)
                    batch_target = batch_target.argmax(1)
                batch_target = batch_target.to(self.device)
                batch_loss = self.loss_criterion(batch_output, batch_target)

                # Backprop
                self.network.zero_grad()
                batch_loss.backward()

                # Update weights
                self.optimizer.step()

            if epoch % (self.nb_epoch / 100) == 0:
                training_loss = self.eval_loss(input_dataset, target_dataset)
                validation_loss = self.eval_loss(input_dataset_val, target_dataset_val)

                self.epochs_w_loss_measure.append(epoch)
                self.training_losses.append(training_loss)
                self.validation_losses.append(validation_loss)

                if self.problem_type == "classification":  # Then also compute accuracies
                    train_preds = self.network.forward(input_dataset)
                    train_preds = train_preds.detach().numpy().argmax(axis=1).squeeze()
                    train_targets = target_dataset.argmax(axis=1).squeeze()
                    training_accuracy = (train_preds == train_targets).mean()
                    self.training_accuracies.append(training_accuracy)

                    val_preds = self.network.forward(input_dataset_val)
                    val_preds = val_preds.detach().numpy().argmax(axis=1).squeeze()
                    val_targets = target_dataset_val.argmax(axis=1).squeeze()
                    validation_accuracy = (val_preds == val_targets).mean()
                    self.validation_accuracies.append(validation_accuracy)

                info = {'training_loss': training_loss, 'validation_loss': validation_loss}

                if self.problem_type == "classification":
                    info['training_accuracy'] = training_accuracy
                    info['validation_accuracy'] = validation_accuracy

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, epoch)

                for tag, value in self.network.named_parameters():
                    tag = tag.replace('.', '/')
                    self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                    self.logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

                if epoch % (self.nb_epoch / 20) == 0:
                    print("==== Loss stats for epoch " + str(epoch) + " ====")
                    print("Training loss = " + str(training_loss))
                    print("Validation loss = " + str(validation_loss))

                    if self.problem_type == "classification":
                        print("Training accuracy = {0:.2%}".format(training_accuracy))
                        print("Validation accuracy = {0:.2%}".format(validation_accuracy))


    def eval_loss(self, input_dataset, target_dataset):
        """
        Calculates loss from the network
        NB: turns off dropout and gradient caching as not required when performing an eval
        """

        with torch.no_grad():  # Stops autograd engine

            # Deactivates dropout layers
            self.network.eval()

            # Forward pass
            output_tensor = self.network.forward(input_dataset)

            # Calculate loss
            if self.problem_type == "regression":
                target_tensor = torch.from_numpy(target_dataset).float()
            elif self.problem_type == "classification":
                # https://discuss.pytorch.org/t/loss-functions-for-batches/20488
                target_tensor = torch.from_numpy(target_dataset).type(torch.long)
                target_tensor = target_tensor.argmax(1)

            target_tensor = target_tensor.to(self.device)
            loss = self.loss_criterion(output_tensor, target_tensor)

            return loss.item()


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


def create_output_folder(run_type):
    """
    Creates an output folder for this run
    """
    # Create a timestamp for saving results
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(timestamp)
    readable_time = str(timestamp.year) + str(timestamp.month).zfill(2) + str(timestamp.day).zfill(2) + "_" + str(timestamp.hour).zfill(2) + str(timestamp.minute).zfill(2) + str(timestamp.second).zfill(2)

    # Create an output folder for this run
    output_path = "output/" + run_type + "/" + readable_time
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Creating new folder: " + output_path)

    return output_path, readable_time



def save_training_output(network, layers, hyper_params, output_path, readable_time,train_loss,val_loss, train_conf = None, val_conf = None):
    """
    Saves training output to file
    """

    # Save pytorch model
    save_torch_model(network, layers, output_path + "/" + readable_time + "_model")

    # Save hyperparameters to log file
    parameter_out_file = output_path + "/parameters.txt"
    with open(parameter_out_file, 'w') as f:
        #f.write("------ LOG FILE ------\n")
        #f.write("Model ran at " + str(readable_time) + "\n\n")

        f.write("Hyperparameters:\n")

        for key, value in hyper_params.items():
            f.write(key + " = " + str(value) + "\n")

        f.write("\n")
        if train_conf != None:
            for key, value in train_conf.items():
                f.write(key + " = " + str(value) + "\n")
            f.write("\n")
            for key, value in val_conf.items():
                f.write(key + " = " + str(value) + "\n")
            f.write("\n")
            """"
        f.write("\n\nLayers:\n")
        for layer in layers:
            if layer.name == "linear":
                f.write(layer.name + "(" + str(layer.in_dim) + ", " + str(layer.out_dim) + ")\n")
            if layer.name == "relu":
                f.write(layer.name + "\n")
            if layer.name == "dropout":
                f.write(layer.name + "(p=" + str(layer.p) +)
                """
        f.write("Training Loss: " + str(train_loss)+"\n")
        f.write("Validation Loss: " + str(val_loss))

    f.close()


def train_fm(is_gpu_run=False):

    # Device configuration
    device = torch.device('cuda' if is_gpu_run else 'cpu')
    print("Running on device " + str(device))

    # Create an output folder for results of this run
    output_path, readable_time = create_output_folder("learn_fm")

    # Load and prepare data
    dataset = np.loadtxt("FM_dataset.dat")
    # Split data
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(dataset, 2)

    # TODO: preprocess the data

    train_prep = TorchPreprocessor(x_train,-1,1)
    x_train_pre = train_prep.apply(x_train)
    x_val_pre = train_prep.apply(x_val)
    x_test_pre = train_prep.apply(x_test)

    # Instatiate a network
    layers = [LinearLayer(name="linear", in_dim=3, out_dim=32),
              # LinearLayer(name="linear", in_dim=8, out_dim=8),
              TanhLayer(name="tanh"),
              # DropoutLayer(name="dropout", p=0.2),
              LinearLayer(name="linear", in_dim=32, out_dim=32),
              TanhLayer(name="tanh"),
              # DropoutLayer(name="dropout", p=0.5),
              # LinearLayer(name="linear", in_dim=64, out_dim=64),
              # TanhLayer(name="tanh"),
              # DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=32, out_dim=3)]

    network = SequentialNet(layers, device)
    print("Network instatiated:")
    print(network)

    # Add the network to a trainer and train
    hyper_params = {'batch_size': 32,
                    'nb_epoch': 10,
                    'learning_rate': 0.005,
                    'loss_fun': "mse",
                    'shuffle_flag': True,
                    'optimizer': "adam"}

    trainer = TorchTrainer(
        problem_type="regression",
        network=network,
        batch_size=hyper_params['batch_size'],
        nb_epoch=hyper_params['nb_epoch'],
        learning_rate=hyper_params['learning_rate'],
        loss_fun=hyper_params['loss_fun'],
        shuffle_flag=hyper_params['shuffle_flag'],
        optimizer=hyper_params['optimizer'],
        device=device,
        log_output_path=output_path
    )

    trainer.train(x_train_pre, y_train, x_val_pre, y_val)

    # Evaluate results
    train_loss=trainer.eval_loss(x_train_pre, y_train)
    val_loss=trainer.eval_loss(x_val_pre, y_val)


    print("Final train loss = {0:.2f}".format(train_loss))
    print("Final validation loss = {0:.2f}".format(val_loss))

        #confusion matrix

    # Save model + hyperparamers to file

class ROIResampler():

    def __init__(self, x_data, y_data, majority_idx, create_synthetic = False):
        self.x_data = x_data
        self.y_data = y_data
        self.full_data = np.concatenate((self.x_data,self.y_data), axis=1)

        if create_synthetic == True:
            self.majority_idx = majority_idx
            self.majority_data = self.full_data[self.full_data[:,self.majority_idx]==1,:]
            self.resampling_data = self.full_data[self.full_data[:,self.majority_idx]!=1,:]

        else:
            self.resampling_data = self.full_data

        self.x_train = self.resampling_data[:,:3]
        self.y_train = self.resampling_data[:,3:]

    def resample(self):
        y_consolidated = np.argmax(self.y_train,axis=1)
        sm = SMOTE(random_state=2)
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_sample(self.x_train,y_consolidated.ravel())

        x_final_training = np.asarray(X_train_res)
        y_final_training = np.asarray(y_train_res)

        b = np.zeros((y_final_training.shape[0],4))
        b[np.arange(y_final_training.shape[0]), y_final_training] = 1
        y_final_training = b

        print(x_final_training.shape)
        print(y_final_training.shape)

        return x_final_training, y_final_training

# TODO: could abstract this further to reduce code repetition
def train_roi(is_gpu_run=False):

    # Device configuration
    device = torch.device('cuda' if is_gpu_run else 'cpu')
    print("Running on device " + str(device))


    # Create an output folder for results of this run
    output_path, readable_time = create_output_folder("learn_roi")

    # Load and prepare data
    dataset = np.loadtxt("ROI_dataset.dat")

    # Split data
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(dataset, 2)

    # TODO: preprocess the data
    x_train_pre = x_train
    x_val_pre = x_val
    x_test_pre = x_test

    resampler = ROIResampler(x_train,y_train,6)
    x_train_res, y_train_res = resampler.resample()


    # Instatiate a network
    layers = [LinearLayer(name="linear", in_dim=3, out_dim=64),
              # LinearLayer(name="linear", in_dim=8, out_dim=8),
              ReluLayer(name="relu"),
              # DropoutLayer(name="dropout", p=0.5),
              # LinearLayer(name="linear", in_dim=8, out_dim=8),
              # ReluLayer(name="relu"),
              # DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=64, out_dim=4),
              SoftmaxLayer(name="softmax")]

    network = SequentialNet(layers, device)
    print("Network instatiated:")
    print(network)

    # Add the network to a trainer and train
    hyper_params = {'batch_size': 32,
                    'nb_epoch': 10,
                    'learning_rate': 0.005,
                    'loss_fun': "cross_entropy",
                    'shuffle_flag': True,
                    'optimizer': "adam"}

    trainer = TorchTrainer(
        problem_type="classification",
        network=network,
        batch_size=hyper_params['batch_size'],
        nb_epoch=hyper_params['nb_epoch'],
        learning_rate=hyper_params['learning_rate'],
        loss_fun=hyper_params['loss_fun'],
        shuffle_flag=hyper_params['shuffle_flag'],
        optimizer=hyper_params['optimizer'],
        device=device,
        log_output_path=output_path
    )

    trainer.train(x_train_res, y_train_res, x_val_pre, y_val)
    train_loss=trainer.eval_loss(x_train_res, y_train_res)
    val_loss=trainer.eval_loss(x_val_pre, y_val)
    # Evaluate results
    print("Final train loss = {0:.2f}".format(train_loss))
    print("Final validation loss = {0:.2f}".format(val_loss))

    print(network.forward(x_train_res))
    print((y_train))

    train_preds = (network.forward(x_train_res)).detach().numpy().argmax(axis=1).squeeze()
    train_targ = y_train_res.argmax(axis=1).squeeze()
   # print(train_preds)
   # print(train_targ)

    conf_matrix=confusion_matrix(train_targ,train_preds)
    recall=recall_calculator(conf_matrix)
    precision=precision_calculator(conf_matrix)
    f1=f1_score_calculator(precision,recall)

    Train_confusion = {
    "Train Confusion Matrix" + "\n" : conf_matrix,
    'Recall': recall,
    'Precision' : precision,
    'F1' : f1}

    val_preds = (network.forward(x_val_pre)).detach().numpy().argmax(axis=1).squeeze()
    val_targ = y_val.argmax(axis=1).squeeze()

    conf_matrix=confusion_matrix(val_targ,val_preds)
    recall=recall_calculator(conf_matrix)
    precision=precision_calculator(conf_matrix)
    f1=f1_score_calculator(precision,recall)

    Val_confusion = {
    "Val Confusion Matrix" + "\n"  : conf_matrix,
    'Recall' : recall,
    'Precision' : precision,
    'F1' : f1}


    # Save model + hyperparamers to file
    save_training_output(network, layers, hyper_params, output_path, readable_time, train_loss, val_loss, Train_confusion, Val_confusion)


if __name__ == "__main__":

    # Get command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvm:", ["model="])
    except getopt.GetoptError:
        print("pytorch_net.py -m <model_name> optional: [-v (verbose output)]")
        sys.exit(2)

    model = None
    for opt, arg in opts:
        if opt == '-h':
            print("pytorch_net.py -m <model_name> optional: [-v (verbose output)]")
            print("-m <model_name>: either learn_fm or learn_roi")
        elif opt in ("-m", "--model"):
            model = arg

    if not model:
        print("error: must provide model -m")
        sys.exit(-1)

    # Run code
    if model == "learn_fm":
        train_fm(is_gpu_run=False)
    elif model == "learn_roi":
        train_roi(is_gpu_run=False)
    else:
        raise ValueError("Not a valid model " + str(model))
