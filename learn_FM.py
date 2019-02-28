import numpy as np

from illustrate import illustrate_results_FM
from evaluation_utils import *
from torch_utils import *


def main(dataset_filepath):

    # Load dataset
    dataset = np.loadtxt(dataset_filepath)

    # Load best model and evaluate
    #model_path = "output/learn_fm/best_model/"
    model_path = "output/learn_fm/20190228_172456/"
    model = load_torch_model(model_path + "model.pt", model_path + "layers.pickle")
    evaluate_architecture(model_path, dataset)

    # Normalise data using training pre-processor
    # TODO:

    # Return predictions to user
    predict_hidden(model, dataset)

    
if __name__ == "__main__":

    # TODO: command line argument
    dataset_filepath = "FM_dataset.dat"
    
    main(dataset_filepath)
