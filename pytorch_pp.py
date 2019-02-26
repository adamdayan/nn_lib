import pprint
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class DataStats(object):

    def __init__(self, data, split_idx, dataset_name, problem_type, plots_folder='plots/'):
        self.split_idx = split_idx
        self.dataset_name = dataset_name
        self.plots_folder = plots_folder

        self.calculate_data_stats(data)
        self.plot_data_box_plot(data)
        if problem_type == "classification":
            self.plot_class_count(data[:, (self.split_idx + 1):])

    def calculate_data_stats(self, data):
        """
        Gives you basic stats on the data you pass in
        """
        self.stats_obj = stats.describe(data)

        # stats_obj is a NamedTuple type
        for key, val in self.stats_obj._asdict().items():
           print(key + ":")
           pprint.pprint(val)

    def plot_data_box_plot(self, data):
        """
        Creates a box plot of the data 
        """
        
        # Box plot of mean, variance and range etc
        # https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51
        plt.figure(figsize=(20,10))
        plt.suptitle('Box plot of features and target columns ' + self.dataset_name)

        plt.subplot(2, 1, 1)
        plt.xlabel("Feature columns")
        red_square = dict(markerfacecolor='r', marker='s')
        plt.boxplot(
            data[:, :(self.split_idx+1)],  # Don't plot the last column
            flierprops=red_square,
            showmeans=True
        )

        plt.subplot(2, 1, 2)
        plt.xlabel("Target columns")
        red_square = dict(markerfacecolor='r', marker='s')
        plt.boxplot(
            data[:, (self.split_idx+1):],  # Don't plot the last column
            flierprops=red_square,
            showmeans=True
        )

        if not os.path.exists(self.plots_folder):
            print("Creating new folder: " + self.plots_folder)
            os.makedirs(self.plots_folder)
        plt.savefig(self.plots_folder + self.dataset_name + "_box_plot.png")

    def plot_class_count(self, target_data):
        """
        Plots a class count of the one hot encoded data
        """
        x = [i+1 for i in range(target_data.shape[1])]
        y = np.sum(target_data, axis=0)

        fig, ax = plt.subplots()

        ax.set_title('Class count: ' + self.dataset_name)
        ax.bar(x, y)

        if not os.path.exists(self.plots_folder):
            print("Creating new folder: " + self.plots_folder)
            os.makedirs(self.plots_folder)
        fig.savefig(self.plots_folder + self.dataset_name + "_class_count.png")


class TorchPreProcessor(object):

    def __init__(self):
        pass


if __name__ == "__main__":    

    print("Plotting stats for FM_dataset.dat")
    data_fm = np.loadtxt("FM_dataset.dat")
    torch_pp_fm = DataStats(data_fm, split_idx=2, dataset_name="FM", problem_type="regression")

    print("Plotting stats for ROI_dataset.dat")
    data_roi = np.loadtxt("ROI_dataset.dat")
    torch_pp_fm = DataStats(data_roi, split_idx=2, dataset_name="ROI", problem_type="regression")