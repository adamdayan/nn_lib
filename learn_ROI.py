import sys
import getopt

import numpy as np

from illustrate import illustrate_results_ROI
from evaluation_utils import *
from torch_utils import *

def main(dataset_filepath):

    # Load dataset
    print("====================================")
    print("Loading dataset = " + dataset_filepath)
    dataset = np.loadtxt(dataset_filepath)

    # Load best model and evaluate
    model_path = "output/learn_roi/best_model/"
    print("====================================")
    print("Loading model = " + model_path)
    print("Model information: ")
    model, x_pp, _ = load_torch_model(model_path + "model.pt",
                                      model_path + "layers.pickle",
                                      model_path + "x_preprocessor.pickle")
    evaluate_architecture(model_path, dataset, "classification")

    # Normalise data using training pre-processor
    dataset = x_pp.apply(dataset[:, :3])
    
    # Return predictions to user
    print("====================================")
    print("Prediction:")
    predict_hidden(model, dataset, x_pp, problem_type="classification")
    

if __name__ == "__main__":

    # Get command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:", ["file="])
    except getopt.GetoptError:
        print("learn_ROI.py -f <filepath>")
        sys.exit(-1)

    dataset_filepath = None
    for opt, arg in opts:
        if opt == '-h':
            print("learn_ROI.py -f <filepath>")
        elif opt in ("-f", "--filepath"):
            dataset_filepath = arg

    if dataset_filepath is None:
        print("error: must provide filepath -f")
        sys.exit(-1)
    
    main(dataset_filepath)
