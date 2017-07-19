
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


class Shapenetcore_Part_Dataset(data.Dataset):
    def __init__(self, data_dir, num_points=2048, mode='train'):
        self.num_points = num_points
        self.data_dir = data_dir
        self.mode = mode
        self.color_map = json.load(open(os.path.join(self.data_dir, 'part_color_mapping.json'), 'r'))
    
        self.categories2folders = {} 
        with open(os.path.join(self.data_dir, 'all_object_categories.txt'), 'r') as f:
            for line in f: 
                ls = line.strip().split() 
                self.categories2folders[ls[0]] = ls[1] 

        print(self.categories2folders) 
        # debug_here() 
        # category
        self.all_cats = json.load(open(os.path.join(self.data_dir, 'overallid_to_catid_partid.json'), 'r'))
        self.num_classes = len(self.categories2folders)
        
        self.num_seg_classes = len(self.all_cats)

        self.train_files_path = os.path.join(self.data_dir, 'train_hdf5_file_list.txt')
        self.val_files_path = os.path.join(self.data_dir, 'val_hdf5_file_list.txt')
        self.test_files_path = os.path.join(self.data_dir, 'test_hdf5_file_list.txt')

        self.train_files_list = [line.rstrip() for line in open(self.train_files_path)]
        self.val_files_list = [line.rstrip() for line in open(self.val_files_path)]
        self.test_files_list = [line.rstrip() for line in open(self.test_files_path)]

        # loading train data 
        if self.mode == 'train':
            print('loading train data ') 
            self.train_data = [] 
            self.train_labels = [] 
            self.train_segs = [] 

            for file_name in self.train_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                segs = file_data['pid'][:]

                self.train_data.append(data)
                self.train_labels.append(labels)
                self.train_segs.append(segs)

            self.train_data = np.concatenate(self.train_data)
            self.train_labels = np.concatenate(self.train_labels)
            self.train_segs = np.concatenate(self.train_segs)

            self.num_points = self.train_data.shape[1] # will be 2048
            # debug_here() 
            # print('hello')
        elif self.mode == 'val': # validation 
            print('loading val data')
            self.val_data = [] 
            self.val_labels = [] 
            self.val_segs = [] 
            for file_name in self.val_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                segs = file_data['pid'][:]
                self.val_data.append(data)
                self.val_labels.append(labels)
                self.val_segs.append(segs)

            self.val_data = np.concatenate(self.val_data)
            self.val_labels = np.concatenate(self.val_labels)
            self.val_segs = np.concatenate(self.val_segs)
            self.num_points = self.val_data.shape[1] 
        else: # test
            # debug_here()
            print('loading test data ') 
            self.test_data = [] 
            self.test_labels = [] 
            self.test_segs = [] 
            for file_name in self.test_files_list:
                file_path = os.path.join(self.data_dir, file_name)
                file_data = h5py.File(file_path)
                data = file_data['data'][:]
                labels = file_data['label'][:]
                segs = file_data['pid'][:]
                self.test_data.append(data)
                self.test_labels.append(labels)
                self.test_segs.append(segs)

            self.test_data = np.concatenate(self.test_data)
            self.test_labels = np.concatenate(self.test_labels)
            self.test_segs = np.concatenate(self.test_segs)
            self.num_points = self.test_data.shape[1] 

    def __getitem__(self, index):
        if self.mode == 'train':
            points, label, segs = self.train_data[index], self.train_labels[index], self.train_segs[index]
        elif self.mode == 'val':
            points, label, segs = self.val_data[index], self.val_labels[index], self.val_segs[index]
        else: # test 
            points, label, segs = self.test_data[index], self.test_labels[index], self.test_segs[index]


        return points, label, segs 

    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        elif self.mode == 'val':
            return self.val_data.shape[0]
        else: 
            return self.test_data.shape[0]


if __name__ == '__main__':
    print('test')
    # debug_here() 
    data = Shapenetcore_Part_Dataset(data_dir='../datasets/shapenet_part_seg_hdf5_dataset', mode='train')
    # debug_here() 

    print(len(data))
    print(data[0])

    data = Shapenetcore_Part_Dataset(data_dir = '../datasets/shapenet_part_seg_hdf5_dataset', mode='val')
    print(len(data))
    points, label, segs = data[0]
    # debug_here()
    data = Shapenetcore_Part_Dataset(data_dir = '../datasets/shapenet_part_seg_hdf5_dataset', mode='test')
    print(len(data))
    points, label, segs = data[0]
    # debug_here()
