from __future__ import print_function 
import argparse
import random
import time
import os
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
parser.add_argument('--data_dir', default='./datasets/shapenet_part_seg_hdf5_dataset', help='path to dataset')
# number of workers for loading data
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# loading data 
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

# on network 
# spcify optimization stuff 
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

parser.add_argument('--max_epochs', type=int, default=140, help='number of epochs to train for')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--print_freq', type=int, default=25, help='number of iterations to print ') 
parser.add_argument('--checkpoint_folder', default=None, help='check point path')
parser.add_argument('--model', type=str, default = '',  help='model path')

# cuda stuff 
parser.add_argument('--gpu_id'  , type=str, default='1', help='which gpu to use, used only when ngpu is 1')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
# clamp parameters into a cube 
parser.add_argument('--gradient_clip', type=float, default=0.01)

# resume training from a checkpoint
parser.add_argument('--init_model', default='', help="model to resume training")
parser.add_argument('--optim_state_from', default='', help="optim state to resume training")

opt = parser.parse_args()
print(opt)

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

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'models_checkpoint'

# make dir 
os.system('mkdir {0}'.format(opt.checkpoint_folder))

# dataset 
if opt.dataset == 'shapenetcore_partanno':
    train_dataset = shapenetcore_partanno_dset.Shapenetcore_Part_Dataset(opt.data_dir, mode='train')
    val_dataset = shapenetcore_partanno_dset.Shapenetcore_Part_Dataset(opt.data_dir, mode='val')
    # we can add test_dataset here
else: 
    print('not supported dataset, so exit')
    exit()

print('number of train samples is: ', len(train_dataset))
print('number of test samples is: ', len(val_dataset))
print('finished loading data')


# input labels: bz x 1 
# output: bz x num_classes is one hot 
def labels_batch2one_hot_batch(labels_batch, num_classes):
    bz = labels_batch.size(0)
    labels_onehot = torch.FloatTensor(bz, num_classes).type_as(labels_batch).zero_()
    labels_onehot.scatter_(1, labels_batch, 1)
    return labels_onehot

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.train() 

    for i, (input_points, _labels, segs) in enumerate(train_loader):
        # bz x 2048 x 3 
        input_points = Variable(input_points)
        input_points = input_points.transpose(2, 1)
        ###############
        ##
        ###############
        _labels = _labels.long() 
        segs = segs.long() 
        labels_onehot = labels_batch2one_hot_batch(_labels, opt.num_classes)
        labels_onehot = Variable(labels_onehot) # we dnonot calculate the gradients here
        # labels_onehot.requires_grad = True
        segs = Variable(segs) 

        if opt.cuda:
            input_points = input_points.cuda() 
            segs = segs.cuda() # must be long cuda tensor 
            labels_onehot = labels_onehot.float().cuda()  # this will be feed into the network
        
        optimizer.zero_grad()
        # forward, backward optimize 
        # pred, _ = model(input_points, labels_onehot)
        pred, _ = model(input_points, labels_onehot)
        pred = pred.view(-1, opt.num_seg_classes)
        segs = segs.view(-1, 1)[:, 0] 
        # debug_here() 
        loss = criterion(pred, segs) 
        loss.backward() 
        ##############################
        # gradient clip stuff 
        ##############################
        utils.clip_gradient(optimizer, opt.gradient_clip)
        optimizer.step() 
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(segs.data).cpu().sum()

        if i % opt.print_freq == 0:
            print('[%d: %d] train loss: %f accuracy: %f' %(i, len(train_loader), loss.data[0], correct/float(opt.batch_size * opt.num_points)))


def validate(val_loader, model, criterion, epoch, opt):
    """Perform validation on the validation set"""
    # switch to evaluate mode
    model.eval()

    top1 = utils.AverageMeter()

    for i, (input_points, _labels, segs) in enumerate(val_loader):
        # bz x 2048 x 3 
        input_points = Variable(input_points, volatile=True)
        input_points = input_points.transpose(2, 1)
        _labels = _labels.long() # this will be feed to the network 
        segs = segs.long()
        labels_onehot = labels_batch2one_hot_batch(_labels, opt.num_classes)
        segs = Variable(segs, volatile=True) 
        labels_onehot = Variable(labels_onehot, volatile=True)

        if opt.cuda:
            input_points = input_points.cuda() 
            segs = segs.cuda() # must be long cuda tensor  
            labels_onehot = labels_onehot.float().cuda() # this will be feed into the network
        
        # forward, backward optimize 
        pred, _ = model(input_points, labels_onehot)
        pred = pred.view(-1, opt.num_seg_classes)
        segs = segs.view(-1, 1)[:, 0]  # min is already 0
        # debug_here() 
        loss = criterion(pred, segs) 

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(segs.data).cpu().sum()

        acc = correct/float(opt.batch_size * opt.num_points)
        top1.update(acc, input_points.size(0))

        if i % opt.print_freq == 0:
            print('[%d: %d] val loss: %f accuracy: %f' %(i, len(val_loader), loss.data[0], acc))
            # print(tested_samples)
    return top1.avg
 
def main():
    global opt
    best_prec1 = 0
    # only used when we resume training from some checkpoint model 
    resume_epoch = 0 
    # train data loader
    # for loader, droplast by default is set to false 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                     shuffle=True, num_workers=int(opt.workers))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                     shuffle=True, num_workers=int(opt.workers))
    
    
    # create model 
    # for modelnet40, opt.num_points is set to be 2048, opt.num_classes is 40
    opt.num_seg_classes = train_dataset.num_seg_classes 
    opt.num_points = train_dataset.num_points
    opt.num_classes = train_dataset.num_classes

    model = pointnet.PointNetPartDenseCls(num_points=opt.num_points, k=opt.num_seg_classes)

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))

        model.load_state_dict(torch.load(opt.init_model))
    # segmentation loss 
    criterion = nn.NLLLoss()

    if opt.cuda:  
        print('shift model and criterion to GPU .. ')
        model = model.cuda() 
        # define loss function (criterion) and pptimizer
        criterion = criterion.cuda()
    # optimizer 

    optimizer = optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    if opt.optim_state_from != '':
        print('loading optim_state_from {0}'.format(opt.optim_state_from))
        optim_state = torch.load(opt.optim_state_from)
        resume_epoch = optim_state['epoch']
        best_prec1 = optim_state['best_prec1']
        # configure optimzer 
        optimizer.load_state_dict(optim_state['optim_state_best'])

    for epoch in range(resume_epoch, opt.max_epochs):
        #################################
        # train for one epoch
        # debug_here()
        #################################
        train(train_loader, model, criterion, optimizer, epoch, opt)


        #################################
        # validate 
        #################################
        prec1 = validate(val_loader, model, criterion, epoch, opt)

        ##################################
        # save checkpoints 
        ################################## 
        if best_prec1 < prec1: 
            best_prec1 = prec1 
            path_checkpoint = '{0}/model_best.pth'.format(opt.checkpoint_folder)
            utils.save_checkpoint(model.state_dict(), path_checkpoint)

            # save optim state 
            path_optim_state = '{0}/optim_state_best.pth'.format(opt.checkpoint_folder)
            optim_state = {} 
            optim_state['epoch'] = epoch + 1 # because epoch starts from 0
            optim_state['best_prec1'] = best_prec1  
            optim_state['optim_state_best'] = optimizer.state_dict() 
            utils.save_checkpoint(optim_state, path_optim_state)
        # problem, should we store latest optim state or model, currently, we donot  
            
        print('best accuracy: ', best_prec1)


if __name__ == '__main__':
    main() 
