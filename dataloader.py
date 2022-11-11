import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.datasets import fetch_openml

class LoadQ1Data:
    def __init__(self, base_path, clause=100, datapoints=500, split='train'):
        '''
        Data Loader class - Read CSV files based on clause and datapoints.
        '''
        self.raw_data = []
        self.label = []
        self.file_format = "{}_c{}_d{}.csv"

        self.data_path = os.path.join(base_path, 
                                      self.file_format.format(
                                        split, clause, datapoints))
        
        self.__readRawData(self.data_path)

    def __readRawData(self, filename):
        '''
        Read each raw data file.
        '''
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                self.label.append(row[-1])
                self.raw_data.append(row[:-1])
        
        self.label = np.array(self.label)
        self.raw_data = np.array(self.raw_data)
    
    def get_data(self):
        return self.raw_data, self.label

    def combine_rawData(self, d1, label1, d2, label2):
        '''
        Combine two datasets to create 1 bigger dataset.
        '''
        d1 = list(d1)
        label1 = list(label1)
    
        d2 = list(d2)
        label2 = list(label2)

        for index in range(len(d2)):
            d1.append(d2[0])
            label1.append(label2[index])
        
        d1 = np.array(d1)
        label1 = np.array(label1)
        return d1, label1

'''
MNIST digit classification dataset - Same as HW2
'''
class MNISTloader:
    def __init__(self, dataset_name = 'mnist_784', num_classes=10, split= 'train'):
        self.num_classes = num_classes

        # load dataset from Sklearn
        X, y = fetch_openml(dataset_name, version=1, return_X_y=True)

        # Scale dataset
        X = X/255

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.get_rawData(X, y)

    def get_rawData(self, X, y):
        X_train = X[:60000]
        X_test = X[60000:]

        Y_train = y[:60000]
        Y_test = y[60000:]

        return X_train.values, Y_train, X_test.values, Y_test
    
    def getData(self):
        if self.split == 'train':
            return self.X_train, self.Y_train
        else:
            return self.X_test, self.Y_test