import time
import datetime
import os
import torch
import pickle
import collections

from pytorch_net import SequentialNet

# Named tuples available globally
LinearLayer = collections.namedtuple("LinearLayer", "name in_dim out_dim")
ReluLayer = collections.namedtuple("ReluLayer", "name")
TanhLayer = collections.namedtuple("TanhLayer", "name")
SoftmaxLayer = collections.namedtuple("SoftmaxLayer", "name")
SigmoidLayer = collections.namedtuple("SigmoidLayer", "name")
DropoutLayer = collections.namedtuple("DropoutLayer", "name p")


def create_output_folder(run_type):
    """
    Creates an output folder for this run
    """
    # Create a timestamp for saving results
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(timestamp)
    readable_time = str(timestamp.year) + str(timestamp.month).zfill(2) + str(timestamp.day).zfill(2) + "_" + str(timestamp.hour).zfill(2) + str(timestamp.minute).zfill(2) + str(timestamp.second).zfill(2)

    # Create an output folder for this run
    output_path = "output/" + run_type + "/" + readable_time + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Creating new folder: " + output_path)

    return output_path, readable_time


def save_training_output(network, layers, hyper_params, output_path, readable_time,train_loss,val_loss, train_conf = None, val_conf = None):
    """
    Saves training output to file
    """

    # Save pytorch model
    save_torch_model(network, layers, output_path)

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


def save_torch_model(model, layers, filepath):
    """
    Saves any pytorch model to file. Use .pt extension
    """
    # Save PyTorch model
    torch.save(model.state_dict(), filepath + "model.pt")

    # Pickle layers config
    with open(filepath + "layers.pickle", "wb") as f:
        pickle.dump(layers, f)

def load_torch_model(model_filename, model_layers_filename):
    """
    Loads a model from a .pt file
    """
    # Load layers config
    with open(model_layers_filename, "rb") as f:
        model_layers = pickle.load(f)

    # Load pytorch model
    model = SequentialNet(layers=model_layers)

    model.load_state_dict(torch.load(model_filename))
    print(model)
    #print(model.state_dict.'bias')


    #print(model)

    # Switch eval mode on for inference
    model.eval()
    return model
