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
import torch.utils.data

from torch.autograd import Variable

import models.pointnet as pointnet
import misc.shapenetcore_partanno_datasets as shapenetcore_partanno_dset
import misc.utils as utils 

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
parser.add_argument('--test_results_dir', type=str, default = None,  help='test path')
parser.add_argument('--output_verbose', type=bool, default=True, help='output verbose')
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
    opt.test_results_dir = 'test_results_folder'

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

def output_color_point_cloud(data, seg, color_map, out_file):
    with open(out_file, 'w') as f:
        l = seg.size(0)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[0][i][0], data[0][i][1], data[0][i][2], color[0], color[1], color[2]))

def output_point_cloud_label_mask(data, seg, out_file):
     with open(out_file, 'w') as f:
        l = seg.size(0)
        for i in range(l):
            f.write('v %f %f %f %f\n' % (data[0][i][0], data[0][i][1], data[0][i][2], seg[i]))



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

            f.write('v %f %f %f %f %f %f\n' % (data[0][i][0], data[0][i][1], data[0][i][2], color[0], color[1], color[2]))

def predict(model, test_loader,color_map, opt):
    ##################################################
    # switch to evaluate mode
    ##################################################
    model.eval()
    ##################################################
    ## log file 
    ##################################################
    # debug_here()
    flog = open(os.path.join(opt.test_results_dir, 'log.txt'), 'w')

    ################Note############################## 
    # each sample may have different number of points 
    # so just use batch of size 1
    ##################################################
    # debug_here() 
    total_acc = 0.0 
    total_seen = 0 
    total_acc_iou = 0.0 
    total_per_label_acc = np.zeros(opt.num_labels).astype(np.float32)
    total_per_label_iou = np.zeros(opt.num_labels).astype(np.float32)
    total_per_label_seen = np.zeros(opt.num_labels).astype(np.int32)
    # currently only support batch size equal to 1 
    for shape_idx, (points_data, _labels, _seg_data) in enumerate(test_loader):
        if shape_idx%10 == 0:
            print('{0}/{1}'.format(shape_idx, len(test_loader)))

        points_data = Variable(points_data, volatile=True)
        points_data = points_data.transpose(2, 1)
        _seg_data = Variable(_seg_data, volatile=True) 
        ##################################################
        ##
        ##################################################
        cur_gt_label = _labels[0][0] 
        cur_label_one_hot = np.zeros((1, opt.num_labels), dtype=np.float32)
        cur_label_one_hot[0, cur_gt_label] = 1
        # ex: [12, 13, 14, 15]
        iou_pids = opt.label_id2pid_set[opt.label_ids[cur_gt_label]]
        # [0, 1, .., 11, 16, ..., 49]
        non_part_labels = list(set(np.arange(opt.num_seg_classes)).difference(set(iou_pids)))
        
        if opt.cuda:
            points_data = points_data.cuda() 
            _seg_data = _seg_data.long().cuda() # must be long cuda tensor  
        
        pred_seg, _ = model(points_data)
        pred_seg = pred_seg.view(-1, opt.num_seg_classes)
        mini = np.min(pred_seg.data.numpy())
        # debug_here()
        pred_seg[:, torch.from_numpy(np.array(non_part_labels))] = mini - 1000
        pred_seg_choice = pred_seg.data.max(1)[1]

        ##################################################################
        ## groundtruth segment mask 
        ##################################################################
        _seg_data = _seg_data.view(-1, 1)[:, 0]  # min is already 0
        
        seg_acc = np.mean(pred_seg_choice.numpy() == _seg_data.data.long().numpy())
        total_acc = seg_acc + total_acc

        total_seen += 1

        total_per_label_seen[cur_gt_label] += 1
        total_per_label_acc[cur_gt_label] += seg_acc
        ############################################
        ##
        ############################################
        mask = np.int32(pred_seg_choice.numpy() == _seg_data.data.long().numpy())
        total_iou = 0.0
        iou_log = ''

        for pid in iou_pids:
            n_pred = np.sum(pred_seg_choice.numpy() == pid)
            n_gt = np.sum(_seg_data.data.long().numpy() == pid)
            n_intersect = np.sum(np.int32(_seg_data.data.long().numpy() == pid) * mask)
            n_union = n_pred + n_gt - n_intersect
            iou_log += '_' + str(n_pred)+'_'+str(n_gt)+'_'+str(n_intersect)+'_'+str(n_union)+'_'
            if n_union == 0:
                total_iou += 1
                iou_log += '_1\n'
            else:
                total_iou += n_intersect * 1.0 / n_union
                iou_log += '_'+str(n_intersect * 1.0 / n_union)+'\n'



        avg_iou = total_iou / len(iou_pids)
        total_acc_iou += avg_iou
        total_per_label_iou[cur_gt_label] += avg_iou
        # debug_here()
        ########################################
        ## transpose data 
        ########################################
        points_data = points_data.transpose(1, 2)
        if opt.output_verbose:
            output_point_cloud_label_mask(points_data.data, _seg_data.data.long(), os.path.join(opt.test_results_dir, str(shape_idx)+'_labels_mask.obj'))

            output_color_point_cloud(points_data.data, _seg_data.data.long(), color_map, os.path.join(opt.test_results_dir, str(shape_idx)+'_gt.obj'))
            output_color_point_cloud(points_data.data, pred_seg_choice, color_map, os.path.join(opt.test_results_dir, str(shape_idx)+'_pred.obj'))
            output_color_point_cloud_red_blue(points_data.data, np.int32(_seg_data.data.long().numpy() == pred_seg_choice.numpy()), 
                    os.path.join(opt.test_results_dir, str(shape_idx)+'_diff.obj'))

            with open(os.path.join(opt.test_results_dir, str(shape_idx)+'.log'), 'w') as fout:
                # fout.write('Total Point: %d\n\n' % ori_point_num)
                fout.write('Ground Truth: %s\n' % opt.label_names[cur_gt_label])
                # fout.write('Predict: %s\n\n' % opt.label_names[label_pred_val])
                fout.write('Accuracy: %f\n' % seg_acc)
                fout.write('IoU: %f\n\n' % avg_iou)
                fout.write('IoU details: %s\n' % iou_log)

        printout(flog, 'Accuracy: %f' % (total_acc / total_seen))
        printout(flog, 'IoU: %f' % (total_acc_iou / total_seen))

        for idx in range(opt.num_labels):
            printout(flog, '\t ' + opt.label_ids[idx] + ' Total Number: ' + str(total_per_label_seen[idx]))
            if total_per_label_acc[idx] > 0:
                printout(flog, '\t ' + opt.label_ids[idx] + ' Accuracy: ' + \
                        str(total_per_label_acc[idx] / total_per_label_acc[idx]))
                printout(flog, '\t ' + opt.label_ids[idx] + ' IoU: '+ \
                        str(total_per_label_iou[idx] / total_per_label_acc[idx]))

        



    print('finished prediction')

def main():
    global opt  

    # part id(1, 2, ..) 
    # for each label_id, there is a set of part id(encoding using 1, 2, 3, 4)
    label_ids2pid = json.load(open(os.path.join(opt.h5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

    label_id2pid_set = {} 
    for idx in range(len(label_ids2pid)): # 50 
        label_id, pid = label_ids2pid[idx] # objid = '02691156'
        if not label_id in label_id2pid_set.keys():
            label_id2pid_set[label_id] = [] 
        label_id2pid_set[label_id].append(idx) # 0, 1, 2, ...
    opt.label_id2pid_set = label_id2pid_set


    all_label_names2label_ids = os.path.join(opt.h5_data_dir, 'all_object_categories.txt')
    fin = open(all_label_names2label_ids, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    # debug_here()
    label_ids = [line.split()[1] for line in lines]
    opt.label_ids = label_ids
    label_names = [line.split()[0] for line in lines]
    opt.label_names = label_names
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
    opt.num_labels = NUM_LABELS

    # 02691156_1: 0 
    # 02691156_2: 1 
    label_id_pid2pid_in_set = json.load(open(os.path.join(opt.h5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

    ####################################################################
    # dataset 
    ####################################################################
    test_dataset = shapenetcore_partanno_dset.Shapenetcore_Part_Dataset(opt.h5_data_dir, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                     shuffle=True, num_workers=int(opt.workers))
    
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

    predict(model, test_loader, color_map, opt) 

if __name__ == '__main__':
    main() 
