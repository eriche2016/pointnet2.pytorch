from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

from IPython.core.debugger import Tracer
debug_here = Tracer()

# transform on raw input data 
# spatial transformer network
class STN3d(nn.Module):
	# for modelnet40, a 3d shape is with 2048 points 
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
    	# x (bz x 3 x 2048) -> conv(3, 64) -> conv(64, 128) -> conv(128, 1024) -> max_pool(2048) -> 1024 -> fc(1024, 512)
    	# 	-> fc(512, 256) -> fc(256, 9)
        
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # bz x 9 
        # identity transform
        # bz x 9 
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) # bz x 3 x 3 
        return x

# 128 x 128 transform
class Feats_STN3d(nn.Module):
    # for modelnet40, a 3d shape is with 2048 points 
    def __init__(self, num_points = 2500):    
        super(Feats_STN3d, self).__init__()
        self.conv1 = nn.Conv1d(128, 256, 1)
        self.conv2 = nn.Conv1d(256, 1024, 1)
        self.mp1 = nn.MaxPool1d(num_points) 

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128*128)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # bz x 256 x 2048 
        x = F.relu(self.bn2(self.conv2(x))) # bz x 1024 x 2048
        x = self.mp1(x) # bz x 1024 x 1
        x = x.view(-1, 1024)

        x = F.relu(self.bn3(self.fc1(x))) # bz x 512 
        x = F.relu(self.bn4(self.fc2(x))) # bz x 256
        x = self.fc3(x) # bz x (128*128) 
        # identity transform
        # bz x (128*128)
        iden = Variable(torch.from_numpy(np.eye(128).astype(np.float32))).view(1,128*128).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 128, 128) # bz x 3 x 3 
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points) # bz x 3 x 3 
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x) # regressing the transforming parameters using STN 
        x = x.transpose(2,1) # bz x 2048 x 3 
        x = torch.bmm(x, trans) # (bz x 2048 x 3) x (bz x 3 x 3) 
        x = x.transpose(2,1) # bz x 3 x 2048
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x # bz x 64 x 2048
        x = F.relu(self.bn2(self.conv2(x))) # bz x 128 x 2048
        x = self.bn3(self.conv3(x)) # bz x 1024 x 2048
        x = self.mp1(x)
        x = x.view(-1, 1024) # bz x 1024
        if self.global_feat: # using global feats for classification
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
	# on modelnet40, it is set to be 2048
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points 
        self.feat = PointNetfeat(num_points, global_feat=True) # bz x 1024 
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x) # bz x 40
        return F.log_softmax(x), trans

# part segmentation 
class PointNetPartDenseCls(nn.Module):
    ###################################
    ## Note that we must use up all the modules defined in __init___, 
    ## otherwise, when gradient clippling, it will cause errors like
    ## param.grad.data.clamp_(-grad_clip, grad_clip)
    ## AttributeError: 'NoneType' object has no attribute 'data'
    ####################################
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetPartDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        # T1 
        self.stn1 = STN3d(num_points = num_points) # bz x 3 x 3, after transform => bz x 2048 x 3 

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # T2 
        self.stn2 = Feats_STN3d(num_points = num_points)

        self.conv4 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 2048, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(2048)
        # pool layer 
        self.mp1 = torch.nn.MaxPool1d(num_points) 

        # MLP(256, 256, 128)
        self.conv7 = torch.nn.Conv1d(3024-16, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 256, 1)
        self.conv9 = torch.nn.Conv1d(256, 128, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)
        # last layer 
        self.conv10 = torch.nn.Conv1d(128, self.k, 1) # 50 
        self.bn10 = nn.BatchNorm1d(self.k)

    def forward(self, x, one_hot_labels):
        batch_size = x.size()[0]
        # T1 
        trans_1 = self.stn1(x) # regressing the transforming parameters using STN
        x = x.transpose(2,1) # bz x 2048 x 3 
        x = torch.bmm(x, trans_1) # (bz x 2048 x 3) x (bz x 3 x 3) 
        # change back 
        x = x.transpose(2,1) # bz x 3 x 2048
        out1 = F.relu(self.bn1(self.conv1(x))) # bz x 64 x 2048
        out2 = F.relu(self.bn2(self.conv2(out1))) # bz x 128 x 2048
        out3 = F.relu(self.bn3(self.conv3(out2))) # bz x 128 x 2048
        #######################################################################
        # T2, currently has bugs so now remove this temporately
        trans_2 = self.stn2(out3) # regressing the transforming parameters using STN
        out3_t = out3.transpose(2,1) # bz x 2048 x 128
        out3_trsf = torch.bmm(out3_t, trans_2) # (bz x 2048 x 128) x (bz x 128 x 3) 
        # change back 
        out3_trsf = out3_trsf.transpose(2,1) # bz x 128 x 2048

        out4 = F.relu(self.bn4(self.conv4(out3_trsf))) # bz x 128 x 2048
        out5 = F.relu(self.bn5(self.conv5(out4))) # bz x 512 x 2048 
        out6 = F.relu(self.bn6(self.conv6(out5))) # bz x 2048 x 2048
        out6 = self.mp1(out6) #  bz x 2048

        # concat out1, out2, ..., out5
        out6 = out6.view(-1, 2048, 1).repeat(1, 1, self.num_points)
        # out6 = x 
        # cetegories is 16
        # one_hot_labels: bz x 16
        one_hot_labels = one_hot_labels.unsqueeze(2).repeat(1, 1, self.num_points)
        # 64 + 128 * 3 + 512 + 2048 + 16  
        # point_feats = torch.cat([out1, out2, out3, out4, out5, out6, one_hot_labels], 1)
        point_feats = torch.cat([out1, out2, out3, out4, out5, out6], 1)
        # Then feed point_feats to MLP(256, 256, 128) 
        mlp = F.relu(self.bn7(self.conv7(point_feats)))
        mlp = F.relu(self.bn8(self.conv8(mlp)))
        mlp = F.relu(self.bn9(self.conv9(mlp)))

        # last layer 
        pred_out = F.relu(self.bn10(self.conv10(mlp))) # bz x 50(self.k) x 2048
        pred_out = pred_out.transpose(2,1).contiguous()
        pred_out = F.log_softmax(pred_out.view(-1,self.k))
        pred_out = pred_out.view(batch_size, self.num_points, self.k)
        return pred_out, trans_1, trans_2

# regular segmentation
class PointNetDenseCls(nn.Module):
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))

    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())
       
    part_seg = PointNetPartDenseCls(k=7)
    one_hot_labels = torch.rand(32, 16)
    debug_here()
    out, _ = part_seg(sim_data, one_hot_labels)
    print('seg', out.size())


