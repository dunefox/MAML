import sys
import struct
from array import array
import numpy as np
import pandas as pd
   
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class IRIS:
    ''' Data class for IRIS data. '''
    
    def __init__(self, file_name):
        '''
        Read the IRIS data which is assumed in the path `path_name`.
        Optionally one can only load a subset of images by specifying
        `num_test` and `num_train`.
        '''

        X_all, Y_all = self.__load_iris_data(file_name)
        self.all_features, self.all_labels  = X_all, Y_all

        # training data
        num_train_samples = int( len(X_all) / 3 )
        self.train_features, self.train_labels  = X_all[:num_train_samples], Y_all[:num_train_samples]

        # data for testing the efficiency
        self.test_features, self.test_labels = X_all[num_train_samples:], Y_all[num_train_samples:]

        # dimension of feature space
        self.dim_features = len(self.train_features[0])
        
        return

    def __load_iris_data(self, file_name):
        ''' 
        Fetches 2d data points from the iris data from the internet archive and
        return (X, Y), where X is a list of 2d points and Y a list of labels.
        '''
       
        # fetch data from internet archive
        df = pd.read_csv(file_name, header=None)
        
        # as feature we take the first two data entries,
        # which are sepal length and width
        X = df.iloc[:, 1:3].values

        # read class labels and convert them to numers as follow:
        # `iris-setosa` set to value -1, `iris-versicol` as well as `iris-virginica` to value 1
        classes = df.iloc[:, 4].values 

        Y = []
        for i in range(len(classes)):
            if classes[i] == 'Iris-setosa':
                Y.append(0)
            elif classes[i] == 'Iris-versicolor':
                Y.append(1)
            elif classes[i] == 'Iris-virginica':
                Y.append(2)
        
        # to make it more realistic, we randomize the data
        indices = np.random.permutation(len(X))
        X_rand = [X[i] for i in indices]
        Y_rand = [Y[i] for i in indices]

        # return the randomized lists as numpy arrays
        return np.array(X_rand), np.array(Y_rand)

