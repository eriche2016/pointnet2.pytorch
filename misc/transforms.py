from __future__ import division
import math
import random
import numpy as np

import torch


class Normalize_PC(object):
    def __init__(self):
        """
        do nothing 
        """
        pass 
    def __call__(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid 
        m = np.max(np.sqrt(np.sum(pc**2, axis=1))) 
        pc = pc/m 

        return pc 

class Augment2PointNum(object):
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, pc):
        assert(pc.shape[0] <= self.num_points)
        cur_len = pc.shape[0]
        res = np.array(pc)
        ###################################
        # copy over and slice
        ###################################
        while cur_len < self.num_points:
            res = np.concatenate((res, pc))
            cur_len += pc.shape[0]

        return res[:self.num_points, :]
        

