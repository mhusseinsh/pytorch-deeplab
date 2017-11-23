from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
from collections import namedtuple
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2

def loggerConfig(log_file, verbose=2):
   logger      = logging.getLogger()
   formatter   = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
   fileHandler = logging.FileHandler(log_file, 'w')
   fileHandler.setFormatter(formatter)
   logger.addHandler(fileHandler)
   if verbose >= 2:
       logger.setLevel(logging.DEBUG)
   elif verbose >= 1:
       logger.setLevel(logging.INFO)
   else:
       # NOTE: we currently use this level to log to get rid of visdom's info printouts
       logger.setLevel(logging.WARNING)
   return logger

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
# NOTE: used as the return format for Env(), and for format to push into replay memory for off-policy methods
# NOTE: when return from Env(), state0 is always None
Experience  = namedtuple('Experience',  'state0, action, reward, state1, terminal1')

# preprocess an image before passing it to a caffe model.
# we need to rescale from [0, 1] to [0, 255], convert from rgb to bgr,
# and subtract the mean pixel.
def preprocess(img_vb):
    img_mean_ts = torch.FloatTensor([103.939, 116.779, 123.68]).cuda() # for vgg
    return img_vb[:, torch.LongTensor([2, 1, 0]).cuda()] * 255. - Variable(img_mean_ts.unsqueeze(1).unsqueeze(2))

# undo the above preprocessing.
def deprocess(img_vb):
    img_mean_ts = torch.FloatTensor([103.939, 116.779, 123.68]).cuda() # for vgg
    img_vb = img_vb + Variable(img_mean_ts.unsqueeze(1).unsqueeze(2))
    return img_vb[:, torch.LongTensor([2, 1, 0]).cuda()] / 255.

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
            stride=stride, padding=1, bias=False)

def outS(k):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    output_list = []
    for i in k:
        j = int(i)
        j = (j+1)//2
        j = int(np.ceil((j+1)/2.0))
        j = (j+1)//2
        output_list.append(j)
    return output_list


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def scale_im(img, scale, interpolation=cv2.INTER_LINEAR):
    new_dims = (int(img.shape[1]*scale),int(img.shape[0]*scale))
    return cv2.resize(img, new_dims, interpolation=interpolation)

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
