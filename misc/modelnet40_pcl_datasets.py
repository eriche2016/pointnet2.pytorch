from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import torch
import json
import h5py

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

import numpy as np
import sys

import json


class Modelnet40_PCL_Dataset(data.Dataset):
    def __init__(self, data_dir, npoints = 2048, train = True):
        self.npoints = npoints
        self.data_dir = data_dir
        self.train = train
        # train files 
        self.train_files_path = os.path.join(self.data_dir, 'train_files.txt')
        self.test_files_path = os.path.join(self.data_dir, 'test_files.txt')
        
        self.train_files_list = [line.rstrip() for line in open(self.train_files_path)]
        self.test_files_list = [line.rstrip() for line in open(self.test_files_path)]
	    # loading train files 
        if self.train:
            print('loading training data ')
            self.train_data = [] 
            self.train_labels = [] 
            for file_name in self.train_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                self.train_data.append(data)
                self.train_labels.append(labels)
            self.train_data = np.concatenate(self.train_data)
            self.train_labels = np.concatenate(self.train_labels)
        else:
            print('loading test data ')
            self.test_data = [] 
            self.test_labels = [] 

            for file_name in self.test_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                self.test_data.append(data)
                self.test_labels.append(labels)
            self.test_data = np.concatenate(self.test_data)
            self.test_labels = np.concatenate(self.test_labels)



    def __getitem__(self, index):
    	if self.train:
    	    points, label = self.train_data[index], self.train_labels[index] 
    	else:
    	    points, label = self.test_data[index], self.test_labels[index]

    	return points, label 


    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

if __name__ == '__main__':
    print('test')
    d = Modelnet40_PCL_Dataset(data_dir='../datasets/modelnet40_ply_hdf5_2048', train=True)
    print(len(d))
    print(d[0])

    points, label = d[0]
    # debug_here()
    print(points)
    print(points.shape)
    print(points.dtype)
    print(label.shape)
    print(label.dtype)

    d = Modelnet40_PCL_Dataset(data_dir = '../datasets/modelnet40_ply_hdf5_2048', train=False)
    print(len(d))
    points, label = d[0]
    print(points)
    print(points.shape)
    print(points.dtype)
    print(label.shape)
    print(label.dtype)
