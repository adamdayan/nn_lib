import pprint
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from imblearn.over_sampling import SMOTE, RandomOverSampler


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


class TorchPreprocessor(object):

    def __init__(self, data, lower_bound, upper_bound, type="feature_scaling"):

        self.type = type

        self.min_x = np.min(data, axis=0)
        self.max_x = np.max(data, axis=0)
        self.lower = lower_bound
        self.upper = upper_bound
        self.mean = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0)

    def apply(self, x):

        if self.type == "normalise":
            return (x - self.mean)/self.stds

        return self.lower + (((x - self.min_x) * (self.upper - self.lower)) / (self.max_x - self.min_x))

    def revert(self, x):

        if self.type == "normalise":
            return (x * self.stds) + self.means

        return self.min_x + (((x - self.lower) * (self.max_x - self.min_x)) / (self.upper - self.lower))


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
    y_test = y_rem[val_idx:]

    print("Input data split into train, val, test with shapes:")
    print("- x_train = " + str(x_train.shape))
    print("- y_train = " + str(y_train.shape))
    print("- x_val = " + str(x_val.shape))
    print("- y_val = " + str(y_val.shape))
    print("- x_test = " + str(x_test.shape))
    print("- y_test = " + str(y_test.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":    

    print("Plotting stats for FM_dataset.dat")
    data_fm = np.loadtxt("FM_dataset.dat")
    torch_pp_fm = DataStats(data_fm, split_idx=2, dataset_name="FM", problem_type="regression")

    print("Plotting stats for ROI_dataset.dat")
    data_roi = np.loadtxt("ROI_dataset.dat")
    torch_pp_fm = DataStats(data_roi, split_idx=2, dataset_name="ROI", problem_type="regression")

