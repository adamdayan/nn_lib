import numpy as np

from illustrate import illustrate_results_ROI
from evaluation_utils import *
from torch_utils import *

def main(dataset_filepath):

    # Load dataset
    dataset = np.loadtxt(dataset_filepath)

    # Load best model and evaluate
    model_path = "output/learn_roi/best_model/"
    model, x_pp, _ = load_torch_model(model_path + "model.pt",
                                      model_path + "layers.pickle",
                                      model_path + "x_preprocessor.pickle")
    evaluate_architecture(model_path, dataset)

    # Normalise data using training pre-processor
    dataset = x_pp.apply(dataset[:, :3])
    
    # Return predictions to user
    predict_hidden(model, dataset, problem_type="classification")
    

if __name__ == "__main__":

    # TODO: command line arguments
    dataset_filepath = "ROI_dataset.dat"
    
    main(dataset_filepath)
