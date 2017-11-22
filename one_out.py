#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Di 21 Nov 2017 23:44:27 CET
Info:
'''

import cv2
import numpy as np
import sys
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from docopt import docopt
import os

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapShot=<str>            Snapshot [default: gta2cityscape184000.pth]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 35]
    --gpu0=<int>                GPU number [default: 0]
    --imgPath=<str>             name of image [default: ../../dataset/gta5/train03_A]
    --savePath=<str>             name of image [default: ../../dataset/gta5/train03_Asemantic]
"""

args = docopt(docstr, version='v0.1')
print args

gpu0 = int(args['--gpu0'])
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
model.eval()
counter = 0
model.cuda(gpu0)
snapShot = args['--snapShot']
saved_state_dict = torch.load(os.path.join('./data/snapshots', snapShot))
interp = nn.UpsamplingBilinear2d(size=(180, 320))
model.load_state_dict(saved_state_dict)

for item in os.listdir(args['--imgPath']):
    read_path = os.path.join(args['--imgPath'], item)
    print read_path
    read_path="/home/tai/test.png"
    img_original = cv2.imread(read_path)
    print img_original.shape
    print img_original[:, 10:20, 0]
    img_temp=img_original.copy()
    img_temp.astype(float)
    img_temp[:,:,0] = img_temp[:,:,0] - 104.008  # B
    img_temp[:,:,1] = img_temp[:,:,1] - 116.669  # G
    img_temp[:,:,2] = img_temp[:,:,2] - 122.675  # R

    output = model(Variable(torch.from_numpy(img_temp[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))
    output = interp(output[3]).cpu().data[0].numpy()
    #output = output[:,:img_temp.shape[0],:img_temp.shape[1]]
    output = output.transpose(1,2,0)
    #output = np.argmax(output,axis = 2)[:, :, np.newaxis].astype(np.uint8)
    #output = np.argmax(output,axis = 2).astype(np.uint8)
    output = np.argmax(output,axis = 2)
    plt.subplot(2, 1, 1)
    plt.imshow(img_original[:,:,::-1])
    plt.subplot(2, 1, 2)
    plt.imshow(output, vmin = 0, vmax = 34)
    plt.show()
    #print output.shape, output.dtype
    #print output
    #img = Image.fromarray(output, 'P')
    #img.save(os.path.join(args['--savePath'], item))



