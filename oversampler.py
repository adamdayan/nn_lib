import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler


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