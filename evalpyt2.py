import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
import caffe
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image

from docopt import docopt

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapShot=<str>            Snapshot [default: gta2cityscape184000.pth]
    --testGTpath=<str>          Ground truth path Shot [default: data/gt/]
    --testIMpath=<str>          Sketch images path Shot [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 35]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')
print args

max_label = int(args['--NoLabels'])-1 # labels from 0,1, ... 20(for VOC)
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print 'pred shape',pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)


        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt)))

    return Aiou

gpu0 = int(args['--gpu0'])
im_path = args['--testIMpath']
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
model.eval()
counter = 0
model.cuda(gpu0)
snapShot = args['--snapShot']
gt_path = args['--testGTpath']
#img_list = open('data/list/train_aug.txt').readlines()
img_list = open('data/list/val.txt').readlines()

#resize_height = 180
#resize_width = 320
resize_height = 360
resize_width = 640

#for iter in range(1,21):   #TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
for iter in range(1):   #TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
    saved_state_dict = torch.load(os.path.join('./data/snapshots/', snapShot))
    if counter==0:
        print snapShot
    counter+=1
    model.load_state_dict(saved_state_dict)

    hist = np.zeros((max_label+1, max_label+1))
    pytorch_list = [];
    for i in img_list:
        img = np.zeros((resize_height,resize_width,3));
        img_original = cv2.imread(os.path.join(im_path,i[:-1]+'.png'))
        new_weidth = int((img_original.shape[1]-(img_original.shape[0]*16//9))//2)
        if new_weidth != 0:
            img_original = img_original[:,new_weidth:-new_weidth]
        img_original = cv2.resize(img_original,(resize_width,resize_height))
        img_temp=img_original.copy()
        img_temp.astype(float)

        img_temp[:,:,0] = img_temp[:,:,0] - 104.008  # B
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669  # G
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675  # R
        img[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        if new_weidth != 0:
            gt = np.array(Image.open(os.path.join(gt_path,i[:-1]+'.png')))[:,new_weidth:-new_weidth]
        #gt[gt==255] = 0
        gt = cv2.resize(gt, (resize_width, resize_height), interpolation = cv2.INTER_NEAREST)
        output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))
        interp = nn.UpsamplingBilinear2d(size=(resize_height, resize_width))
        output = interp(output[3]).cpu().data[0].numpy()
        output = output[:,:img_temp.shape[0],:img_temp.shape[1]]

        output = output.transpose(1,2,0)
        output = np.argmax(output,axis = 2)
        if args['--visualize']:
            plt.subplot(3, 1, 1)
            plt.imshow(img_original[:,:,::-1])
            plt.subplot(3, 1, 2)
            plt.imshow(gt, vmin = 0, vmax = 34)
            plt.subplot(3, 1, 3)
            plt.imshow(output, vmin = 0, vmax = 34)
            plt.show()

        iou_pytorch = get_iou(output,gt)
        pytorch_list.append(iou_pytorch)
        hist_ = fast_hist(gt.flatten(),output.flatten(),max_label+1)
        hist += hist_
        print iou_pytorch, hist
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print 'pytorch',iter,"Mean iou = ",np.sum(miou)/len(pytorch_list)
