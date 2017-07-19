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
import misc.modelnet40_pcl_datasets as modelnet40_dset
import misc.utils as utils 

# import my_modules.utils as mutils

from IPython.core.debugger import Tracer 
debug_here = Tracer() 


parser = argparse.ArgumentParser()

# specify data and datapath 
parser.add_argument('--dataset',  default='modelnet40_pcl', help='modelnet40_pcl | ?? ')
parser.add_argument('--data_dir', default='./datasets/modelnet40_ply_hdf5_2048', help='path to dataset')
# number of workers for loading data
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# loading data 
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
# parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
# parser.add_argument('--nc', type=int, default=3, help='input image channels')
# spicify noise dimension to the Generator 

# on network 
parser.add_argument('--num_classes', type=int, default=40, help='number of classes')
parser.add_argument('--num_points', type=int, default=2048, help='number of points per example')
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

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'models_checkpoint'

# make dir 
os.system('mkdir {0}'.format(opt.checkpoint_folder))

# dataset 
if opt.dataset == 'modelnet40_pcl':
	train_dataset = modelnet40_dset.Modelnet40_PCL_Dataset(opt.data_dir, npoints=2048, train=True)
	test_dataset = modelnet40_dset.Modelnet40_PCL_Dataset(opt.data_dir, npoints=2048, train=False)
else: 
	print('not supported dataset, so exit')
	exit()

print('number of train samples is: ', len(train_dataset))
print('number of test samples is: ', len(test_dataset))
print('finished loading data')

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


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter() 
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter() 

    # training mode
    model.train() 

    end = time.time() 
    for i, (input_points, labels) in enumerate(train_loader):
        # bz x 2048 x 3 
        input_points = Variable(input_points)
        input_points = input_points.transpose(2, 1) 
        labels = Variable(labels[:, 0])

        # print(points.size())
        # print(labels.size())
        # shift data to GPU
        if opt.cuda:
            input_points = input_points.cuda() 
            labels = labels.long().cuda() # must be long cuda tensor  
        
        # forward, backward optimize 
        output, _ = model(input_points)
        # debug_here() 
        loss = criterion(output, labels)
        ##############################
        # measure accuracy
        ##############################
        prec1 = utils.accuracy(output.data, labels.data, topk=(1,))[0]
        losses.update(loss.data[0], input_points.size(0))
        top1.update(prec1[0], input_points.size(0))

        ##############################
        # compute gradient and do sgd 
        ##############################
        optimizer.zero_grad() 
        loss.backward() 
        ##############################
        # gradient clip stuff 
        ##############################
        utils.clip_gradient(optimizer, opt.gradient_clip)
        
        optimizer.step() 

        # measure elapsed time
        batch_time.update(time.time() - end) 
        end = time.time() 
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  loss=losses, top1=top1)) 


def validate(test_loader, model, criterion, epoch, opt):
    """Perform validation on the validation set"""
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # tested_samples = 0 
    for i, (input_points, labels) in enumerate(test_loader):
        # tested_samples = tested_samples + input_points.size(0)

        if opt.cuda:
            input_points = input_points.cuda()
            labels = labels.long().cuda(async=True)
        input_points = input_points.transpose(2, 1)
        input_var = Variable(input_points, volatile=True)
        target_var =  Variable(labels[:, 0], volatile=True)

        # compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target_var.data, topk=(1,))[0]
        losses.update(loss.data[0], input_points.size(0))
        top1.update(prec1[0], input_points.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(test_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # print(tested_samples)
    return top1.avg
 
def main():
    global opt
    best_prec1 = 0
    # only used when we resume training from some checkpoint model 
    resume_epoch = 0 
    # train data loader
    # for loader, droplast by default is set to false 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))
    
    
    # create model 
    # for modelnet40, opt.num_points is set to be 2048, opt.num_classes is 40
    model = pointnet.PointNetCls(num_points = opt.num_points, k = opt.num_classes)
    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    criterion = nn.CrossEntropyLoss()

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
        prec1 = validate(test_loader, model, criterion, epoch, opt)

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
