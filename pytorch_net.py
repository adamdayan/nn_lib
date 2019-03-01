import collections
import sys
import getopt

from logger import Logger
import numpy as np
from skopt import gp_minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from pytorch_pp import *
from sklearn.metrics import confusion_matrix
from torch_utils import *
from evaluation_utils import *
from scipy.special import softmax

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
                 log_output_path='',
                 optimise_flag=False):

        self.problem_type = problem_type
        self.network = network.to(device)
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag
        self.device = device
        self.log_output_path = log_output_path + '/logs'
        if not optimise_flag:
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

    def minibatch_data(self, input_dataset, target_dataset, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        assert input_dataset.shape[0] == target_dataset.shape[0], "input and target dataset do not have same number of datapoints!"

        cut_points = np.arange(batch_size, input_dataset.shape[0], batch_size)
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


    def set_optimisation(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def optimise_hyperparameters(self, params):
        #extract params from input array
        batch_size = params[0]
        learning_rate = params[1]

        input_dataset = self.x_train
        target_dataset = self.y_train
        input_dataset_val = self.x_val
        target_dataset_val = self.y_val
        hidden_layer_neurons = params[2]

        if self.problem_type == "regression":
            #build network
            layers = [LinearLayer(name="linear", in_dim=3, out_dim=hidden_layer_neurons),
                      # LinearLayer(name="linear", in_dim=8, out_dim=8),
                      ReluLayer(name="relu"),
                      # DropoutLayer(name="dropout", p=0.2),
                      LinearLayer(name="linear", in_dim=hidden_layer_neurons, out_dim=hidden_layer_neurons),
                      ReluLayer(name="relu"),
                      # DropoutLayer(name="dropout", p=0.5),
                      LinearLayer(name="linear", in_dim=hidden_layer_neurons, out_dim=3)]

        elif self.problem_type == "classification":
            layers = [LinearLayer(name="linear", in_dim=3, out_dim=64),
                      # LinearLayer(name="linear", in_dim=8, out_dim=8),
                      ReluLayer(name="relu"),
                      # DropoutLayer(name="dropout", p=0.5),
                      # LinearLayer(name="linear", in_dim=8, out_dim=8),
                      # ReluLayer(name="relu"),
                      # DropoutLayer(name="dropout", p=0.5),
                      LinearLayer(name="linear", in_dim=64, out_dim=4),
                      SoftmaxLayer(name="softmax")]

        network = SequentialNet(layers, self.device)
        optimiser = optim.Adam(network.parameters(), lr=learning_rate)

        #split data
        input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
        batched_input, batched_target = self.minibatch_data(input_dataset, target_dataset, batch_size)

        #train model
        for epoch in range(1, self.nb_epoch + 1):
            network.train()

            for batch_input, batch_target in zip(batched_input, batched_target):
                batch_output = network.forward(batch_input)

                if self.problem_type == "regression":
                    cur_batch_target = torch.from_numpy(batch_target).float()
                elif self.problem_type == "classification":
                    cur_batch_target = torch.from_numpy(batch_target).type(torch.long)
                    cur_batch_target = cur_batch_target.argmax(1)

                batch_loss = self.loss_criterion(batch_output, cur_batch_target)

                network.zero_grad()
                batch_loss.backward()
                optimiser.step()

        #evaluate model on val data
        with torch.no_grad():
            output_tensor = network.forward(input_dataset_val)

            if self.problem_type == "regression":
                target_tensor = torch.from_numpy(target_dataset_val).float()
            elif self.problem_type == "classification":
                target_tensor = torch.from_numpy(target_dataset_val).type(torch.long)

                target_tensor = target_tensor.argmax(1)

            validation_loss = self.loss_criterion(output_tensor, target_tensor)

            return validation_loss.item()



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

    # Preprocess the features
    x_preproc = TorchPreprocessor(x_train, -1, 1)
    x_train_pre = x_preproc.apply(x_train)
    x_val_pre = x_preproc.apply(x_val)
    x_test_pre = x_preproc.apply(x_test)

    # Also preprocess the targets for regression
    y_preproc = TorchPreprocessor(y_train, -1, 1, "normalise")
    y_train_pre = y_preproc.apply(y_train)
    y_val_pre = y_preproc.apply(y_val)
    y_test_pre = y_preproc.apply(y_test)

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
    print("Network instantiated:")
    print(network)

    # Add the network to a trainer and train
    hyper_params = {'batch_size': 32,
                    'nb_epoch': 1000,
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

    trainer.train(x_train_pre, y_train_pre, x_val_pre, y_val_pre)

    # Evaluate results
    train_loss=trainer.eval_loss(x_train_pre, y_train_pre)
    val_loss=trainer.eval_loss(x_val_pre, y_val_pre)
    test_loss=trainer.eval_loss(x_test_pre, y_test_pre)
    losses={" Train Loss": train_loss, "Validation Loss": val_loss, "Test Loss":test_loss}

    # Save model + hyperparamers to file
    print("Final train loss = {0:.8f}".format(train_loss))
    print("Final validation loss = {0:.8f}".format(val_loss))
    print("Final test loss = {0:.8f}".format(test_loss))

    save_training_output(network,
                         layers,
                         x_preproc,
                         hyper_params,
                         output_path,
                         readable_time,
                         train_loss,
                         val_loss,
			 test_loss,
                         y_preprocessor=y_preproc)

    # Plot learning curves
    # to check how well model is training (e.g. is there overfitting)
    plt.figure(figsize=(20,10))
    plt.suptitle("Loss vs epochs for " + hyper_params['loss_fun'])

    # Description of the hyperparams
    hyperparams_text = "Hyperparameters: \n " + \
                       "- lr = " + str(hyper_params['learning_rate']) + \
                       "\n - batch_size = " + str(hyper_params['batch_size']) + \
                       "\n - loss function = " + hyper_params['loss_fun'] + \
                       "\n - number of epochs = " + str(hyper_params['nb_epoch'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.75, 0.75, hyperparams_text, bbox=props)

    # Plot loss vs number of epochs
    plt.plot(trainer.epochs_w_loss_measure, trainer.training_losses)
    plt.plot(trainer.epochs_w_loss_measure, trainer.validation_losses)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss (" + hyper_params['loss_fun'] + ")")

    plt.savefig(output_path + "/" + hyper_params['loss_fun'] + "_loss_plot.png")


def confusion_matrices(train_preds,train_targ,val_preds,val_targ,test_preds,test_targ):

        #Training data confusion matrix
        conf_matrix=confusion_matrix(train_targ,train_preds)
        recall=recall_calculator(conf_matrix)
        precision=precision_calculator(conf_matrix)
        f1=f1_score_calculator(precision,recall)

        Train_confusion = {
        "Train Confusion Matrix" + "\n" : conf_matrix,
        'Recall': recall,
        'Precision' : precision,
        'F1' : f1}



        #Validation data confusion matrix
        conf_matrix=confusion_matrix(val_targ,val_preds)
        recall=recall_calculator(conf_matrix)
        precision=precision_calculator(conf_matrix)
        f1=f1_score_calculator(precision,recall)

        Val_confusion = {
        "Val Confusion Matrix" + "\n"  : conf_matrix,
        'Recall' : recall,
        'Precision' : precision,
        'F1' : f1}

        #Test data confusion matrix
        conf_matrix=confusion_matrix(test_targ,test_preds)
        recall=recall_calculator(conf_matrix)
        precision=precision_calculator(conf_matrix)
        f1=f1_score_calculator(precision,recall)

        Test_confusion = {
        "Test Confusion Matrix" + "\n"  : conf_matrix,
        'Recall' : recall,
        'Precision' : precision,
        'F1' : f1}

        return Train_confusion, Val_confusion, Test_confusion




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

    # Preprocess the data
    train_prep = TorchPreprocessor(x_train,-1,1)
    x_train_pre = train_prep.apply(x_train)
    x_val_pre = train_prep.apply(x_val)
    x_test_pre = train_prep.apply(x_test)

    resampler = ROIResampler(x_train_pre,y_train,6)
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
                    'nb_epoch': 50,
                    'learning_rate': 0.001,
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
    train_loss = trainer.eval_loss(x_train_res, y_train_res)
    val_loss = trainer.eval_loss(x_val_pre, y_val)
    test_loss = trainer.eval_loss(x_test_pre, y_test)

    # Evaluate results
    print("Final train loss = {0:.8f}".format(train_loss))
    print("Final validation loss = {0:.8f}".format(val_loss))
    print("Final test loss = {0:.8f}".format(test_loss))

    train_preds = (network.forward(x_train_res)).detach().numpy()
    train_preds = softmax(train_preds).argmax(axis=1).squeeze()
    train_targ = y_train_res.argmax(axis=1).squeeze()

    val_preds = (network.forward(x_val_pre)).detach().numpy()
    val_preds = softmax(val_preds).argmax(axis=1).squeeze()
    val_targ = y_val.argmax(axis=1).squeeze()

    test_preds = (network.forward(x_test_pre)).detach().numpy()
    test_preds = softmax(test_preds).argmax(axis=1).squeeze()
    test_targ = y_test.argmax(axis=1).squeeze()

    train_accuracy = (train_preds == train_targ).mean()
    val_accuracy = (val_preds == val_targ).mean()
    test_accuracy = (test_preds == test_targ).mean()

    losses={"Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Test Loss": test_loss,
            "Train Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Test Accuracy": test_accuracy}

    Train_confusion, Val_confusion, Test_confusion = confusion_matrices(train_preds,train_targ,val_preds,val_targ,test_preds,test_targ)

    # Save model + hyperparamers to file
    save_training_output(network,
                         layers,
                         train_prep,
                         hyper_params,
                         output_path,
                         readable_time,
                         losses,
                         Train_confusion,
                         Val_confusion,
                         Test_confusion)

    # Plot loss and accuracy vs number of epochs
    # to check how well model is training (e.g. is there overfitting)
    plt.figure(figsize=(20,10))
    plt.suptitle("Loss and accuracy vs epochs for " + hyper_params['loss_fun'])
    # Description of the hyperparams
    hyperparams_text = "Hyperparameters: \n " + \
                 "- lr = " + str(hyper_params['learning_rate']) + \
                 "\n - batch_size = " + str(hyper_params['batch_size']) + \
                 "\n - loss function = " + hyper_params['loss_fun'] + \
                 "\n - number of epochs = " + str(hyper_params['nb_epoch'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.75, 0.75, hyperparams_text, bbox=props)

    # Plot loss vs number of epochs
    plt.subplot(2, 1, 1)
    plt.plot(trainer.epochs_w_loss_measure, trainer.training_losses)
    plt.plot(trainer.epochs_w_loss_measure, trainer.validation_losses)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss (" + hyper_params['loss_fun'] + ")")

    # Plot accuracy vs number of epochs
    plt.subplot(2, 1, 2)
    plt.plot(trainer.epochs_w_loss_measure, trainer.training_accuracies)
    plt.plot(trainer.epochs_w_loss_measure, trainer.validation_accuracies)
    plt.legend(['training', 'validation'], loc='upper left')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy (" + hyper_params['loss_fun'] + ")")
    plt.ylim(0, 1)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
    plt.savefig(output_path + "/" + hyper_params['loss_fun'] + "_loss_plot.png")



def optimise_fm():
    device = "cpu"

    # Load and prepare data
    dataset = np.loadtxt("FM_dataset.dat")

    # Split data
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(dataset, 2)

    # TODO: preprocess the data
    # Preprocess the features
    x_preproc = TorchPreprocessor(x_train,-1,1)
    x_train_pre = x_preproc.apply(x_train)
    x_val_pre = x_preproc.apply(x_val)
    x_test_pre = x_preproc.apply(x_test)

    # Also preprocess the targets for regression
    y_preproc = TorchPreprocessor(y_train, -1, 1, "normalise")
    y_train_pre = y_preproc.apply(y_train)
    y_val_pre = y_preproc.apply(y_val)
    y_test_pre = y_preproc.apply(y_test)

    network = SequentialNet([LinearLayer(name="linear", in_dim=3, out_dim=32)])
    trainer = TorchTrainer(
        problem_type="regression",
        network=network,
        batch_size=1000,
        nb_epoch=900,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True,
        optimizer="adam",
        device="cpu",
        log_output_path="",
        optimise_flag=True
    )

    trainer.set_optimisation(x_train_pre, y_train_pre, x_val_pre, y_val_pre)

    optimisation_parameters = [
        (10,250),
        (0.0005, 0.2),
        (10,200)
    ]

    result = gp_minimize(trainer.optimise_hyperparameters,
                         optimisation_parameters,
                         acq_func="EI",
                         n_calls=50,
                         n_random_starts=5,
                         noise=0.1**2,
                         random_state=123
                         )

    #from skopt.plots import plot_convergence

    #plot_convergence(result)

    print("Training model on optimal parameters")
    print(result)
    optimal_parameters_list = result.get("x")
    output_path, readable_time = create_output_folder("best_fm")

    # Instatiate a network
    layers = [LinearLayer(name="linear", in_dim=3, out_dim=optimal_parameters_list[2]),
              # LinearLayer(name="linear", in_dim=8, out_dim=8),
              ReluLayer(name="relu"),
              # DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=optimal_parameters_list[2], out_dim=optimal_parameters_list[2]),

              ReluLayer(name="relu"),
              # DropoutLayer(name="dropout", p=0.5),
              LinearLayer(name="linear", in_dim=optimal_parameters_list[2], out_dim=3),
              #SoftmaxLayer(name="softmax")
    ]

    network = SequentialNet(layers, device)
    print("Optimised Network instatiated:")
    print(network)

    # Add the network to a trainer and train
    hyper_params = {'batch_size': optimal_parameters_list[0],
                    'nb_epoch': 10,
                    'learning_rate': optimal_parameters_list[1],
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

    trainer.train(x_train_pre, y_train_pre, x_val_pre, y_val_pre)

    # Evaluate results
    train_loss = trainer.eval_loss(x_train_pre, y_train_pre)
    val_loss = trainer.eval_loss(x_val_pre, y_val_pre)
    test_loss = trainer.eval_loss(x_test_pre, y_test_pre)
    losses={" Train Loss": train_loss, "Validation Loss": val_loss, "Test Loss":test_loss }

    print("Optimised train loss = {0:.2f}".format(train_loss))
    print("Optimised validation loss = {0:.2f}".format(val_loss))
    print("Optimised test loss = {0:.2f}".format(test_loss))

    # Save model + hyperparamers to file
    save_training_output(network,
                         layers,
                         x_preproc,
                         hyper_params,
                         output_path,
                         readable_time,
                         losses,
                         y_preprocessor=y_preproc)


def optimise_roi():
    device = "cpu"

    # Load and prepare data
    dataset = np.loadtxt("ROI_dataset.dat")

    # Split data
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(dataset, 2)

    # TODO: preprocess the data
    train_prep = TorchPreprocessor(x_train,-1,1)
    x_train_pre = train_prep.apply(x_train)
    x_val_pre = train_prep.apply(x_val)
    x_test_pre = train_prep.apply(x_test)

    resampler = ROIResampler(x_train_pre, y_train, 6)
    x_train_res, y_train_res = resampler.resample()

    network = SequentialNet([LinearLayer(name="linear", in_dim=3, out_dim=32)])
    trainer = TorchTrainer(
        problem_type="classification",
        network=network,
        batch_size=10,
        nb_epoch=200,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
        optimizer="adam",
        device="cpu",
        log_output_path="",
        optimise_flag=True
    )

    trainer.set_optimisation(x_train_res, y_train_res, x_val_pre, y_val)

    optimisation_parameters = [
        (10,200),
        (0.001, 0.2),
        (10,100)]



    result = gp_minimize(trainer.optimise_hyperparameters,
                         optimisation_parameters,
                         acq_func="EI",
                         n_calls=20,
                         n_random_starts=5,
                         noise=0.1**2,
                         random_state=123
                         )

    #from skopt.plots import plot_convergence

    #plot_convergence(result)

    print("Training model on optimal parameters")
    optimal_parameters_list = result.get("x")
    output_path, readable_time = create_output_folder("best_roi")

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
    print("Optimised Network instantiated:")
    print(network)

    # Add the network to a trainer and train
    hyper_params = {'batch_size': optimal_parameters_list[0],
                    'nb_epoch': 250,
                    'learning_rate': optimal_parameters_list[1],
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

    train_loss = trainer.eval_loss(x_train_res, y_train_res)
    val_loss = trainer.eval_loss(x_val_pre, y_val)
    test_loss = trainer.eval_loss(x_test_pre, y_test)

    # Evaluate results
    print("Optimised train loss = {0:.8f}".format(train_loss))
    print("Optimised validation loss = {0:.8f}".format(val_loss))
    print("Optimised test loss = {0:.8f}".format(test_loss))


    train_preds = (network.forward(x_train_res)).detach().numpy()
    train_preds = softmax(train_preds).argmax(axis=1).squeeze()
    train_targ = y_train_res.argmax(axis=1).squeeze()

    val_preds = (network.forward(x_val_pre)).detach().numpy()
    val_preds = softmax(val_preds).argmax(axis=1).squeeze()
    val_targ = y_val.argmax(axis=1).squeeze()

    test_preds = (network.forward(x_test_pre)).detach().numpy()
    test_preds = softmax(val_preds).argmax(axis=1).squeeze()
    test_targ = y_test.argmax(axis=1).squeeze()

    Train_confusion, Val_confusion, Test_confusion = confusion_matrices(train_preds,train_targ,val_preds,val_targ,test_preds,test_targ)

    losses={" Train Loss": train_loss, "Validation Loss": val_loss, "Test Loss":test_loss }

    # Save model + hyperparamers to file
    save_training_output(network, layers, train_prep, hyper_params, output_path, readable_time,losses, Train_confusion, Val_confusion, Test_confusion)


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
    elif model == "optimise_fm":
        optimise_fm()
    elif model == "optimise_roi":
        optimise_roi()
    else:
        raise ValueError("Not a valid model " + str(model))
