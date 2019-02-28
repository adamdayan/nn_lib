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



def save_torch_model(model, layers, filename):
    """
    Saves any pytorch model to file. Use .pt extension
    """
    # Save PyTorch model
    torch.save(model.state_dict(), filename + "_model.pt")

    # Pickle layers config
    with open(filename + "_layers.pickle", "wb") as f:
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
