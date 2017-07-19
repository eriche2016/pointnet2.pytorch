from __future__ import print_function 
import argparse
import random
import time
import os
import json 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.parallel # for multi-GPU training 
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms # using transforms 
import torch.utils.data

from torch.autograd import Variable

import models.pointnet as pointnet
import misc.shapenetcore_partanno_datasets as shapenetcore_partanno_dset
import misc.utils as utils 

import misc.transforms as pc_transforms 

import misc.shapenet_test_from_list as shpnt_t_frm_lst

# import my_modules.utils as mutils

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

parser = argparse.ArgumentParser()

# specify data and datapath 
parser.add_argument('--dataset',  default='shapenetcore_partanno', help='shapenetcore_partanno | ?? ')
# ply data dir 
parser.add_argument('--ply_data_dir', default='./datasets/raw_datasets/PartAnnotation', help='path to ply data')
parser.add_argument('--h5_data_dir', default='./datasets/shapenet_part_seg_hdf5_dataset', help='path to h5 data')
# number of workers for loading data
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# loading data 
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')

parser.add_argument('--print_freq', type=int, default=25, help='number of iterations to print ') 
parser.add_argument('--pretrained_model', type=str, default = './models_checkpoint/model_best.pth',  help='model path')
parser.add_argument('--test_results_dir', type=str, default = None,  help='model path')

# cuda stuff 
parser.add_argument('--gpu_id'  , type=str, default='1', help='which gpu to use, used only when ngpu is 1')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')

#####################################################################
## global setting 
#####################################################################
opt = parser.parse_args()
print(opt)
if opt.test_results_dir is None:
    opt.test_results_folder = 'test_results_folder'

# make dir 
os.system('mkdir {0}'.format(opt.test_results_dir))


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

ngpu = int(opt.ngpu)
# opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 123456

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if ngpu == 1: 
        print('so we use 1 gpu to training') 
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
##############################################################
##
##############################################################
def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def predict(model, test_loader):
    # switch to evaluate mode
    model.eval()

    ################Note############################## 
    # each sample may have different number of points 
    # so just use batch of size 1
    ##################################################
    debug_here() 
    for i, (points_data, _seg_data, labels) in enumerate(test_loader, 0):
        if i%10 == 0:
            print('{0}/{1}'.format(i, len(test_loader)))
            # print(points_data.size())

        points_data = Variable(points_data, volatile=True)
        points_data = points_data.transpose(2, 1)
        _seg_data = Variable(_seg_data, volatile=True) 

        if opt.cuda:
            points_data = points_data.cuda() 
            _seg_data = _seg_data.long().cuda() # must be long cuda tensor  
        
        # forward, backward optimize 
        pred, _ = model(points_data)
        pred = pred.view(-1, opt.num_seg_classes)
        _seg_data = _seg_data.view(-1, 1)[:, 0]  # min is already 0
        pred_choice = pred.data.max(1)[1]

    print('finished loading')

def main():
    global opt  
    ################################################
    # should specify it to be 3000??????????????
    ################################################
    MAX_NUM_POINTS = 3000         # the max number of points in the all testing data shapes

    pc_transform_all = transforms.Compose([
    pc_transforms.Normalize_PC(),
    pc_transforms.Augment2PointNum(MAX_NUM_POINTS),
    ])


    # part id(1, 2, ..) 
    # for each label_id, there is a set of part id(encoding using 1, 2, 3, 4)
    label_ids2pid = json.load(open(os.path.join(opt.h5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

    label_id2pid_set = {} 
    for idx in range(len(label_ids2pid)): # 50 
        label_id, pid = label_ids2pid[idx] # objid = '02691156'
        if not label_id in label_id2pid_set.keys():
            label_id2pid_set[label_id] = [] 
        label_id2pid_set[label_id].append(idx) # 0, 1, 2, ...


    all_label_names2label_ids = os.path.join(opt.h5_data_dir, 'all_object_categories.txt')
    fin = open(all_label_names2label_ids, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    debug_here()
    label_ids = [line.split()[1] for line in lines]
    label_names = [line.split()[0] for line in lines]
    # 
    label_ids2ids = {label_ids[i]:i for i in range(len(label_ids))}
    fin.close()

    color_map_file_path = os.path.join(opt.h5_data_dir, 'part_color_mapping.json')
    # 50 color map 
    # color_map[0] = [0.65, 0.95, 0.05]
    color_map = json.load(open(color_map_file_path, 'r'))
    NUM_LABELS = len(label_ids) # 16
    NUM_PARTS = len(label_ids2pid)  # 50
    opt.num_seg_classes = NUM_PARTS 

    # 02691156_1: 0 
    # 02691156_2: 1 
    label_id_pid2pid_in_set = json.load(open(os.path.join(opt.h5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

    # call predict 
    test_ply_data_list_path = os.path.join(opt.ply_data_dir, 'test_ply_file_list.txt')

    test_dataset = shpnt_t_frm_lst.PlyFileList(opt.ply_data_dir, test_ply_data_list_path, 
        label_id_pid2pid_in_set, label_ids2ids, label_ids, transform=pc_transform_all)

    ################Note############################## 
    # each sample may have different number of points 
    # so just use batch of size 1
    ##################################################
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    ########################################################################
    ##
    ########################################################################
    assert opt.pretrained_model != '', 'must specify the pre-trained model'
    print("Loading Pretrained Model from {0}".format(opt.pretrained_model))
    model = pointnet.PointNetDenseCls(num_points=opt.num_points, k=opt.num_seg_classes)
    model.load_state_dict(torch.load(opt.pretrained_model))

    if opt.cuda:  
        print('shift model and criterion to GPU .. ')
        model = model.cuda()

    predict(model, test_loader) 

if __name__ == '__main__':
    main() 
