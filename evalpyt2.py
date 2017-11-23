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
import random

from docopt import docopt

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapShot=<str>            Snapshot [default: g2c1122_93000.pth]
    --testGTpath=<str>          Ground truth path Shot [default: data/train03B]
    --testIMpath=<str>          Sketch images path Shot [default: data/train03B]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 35]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')
print args

max_label = int(args['--NoLabels'])-1 # labels from 0,1, ... 20(for VOC)
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def outS(k):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    output_list = []
    for i in k:
        j = int(i)
        j = (j+1)/2
        j = int(np.ceil((j+1)/2.0))
        j = (j+1)/2
        output_list.append(j)
    return output_list

def resize_label_batch(label, size):
    label_resized = np.zeros((size[0],size[1],1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size[0], size[1]))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

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
img_list = open('./data/list/trainblist.txt').readlines()

#resize_height = 180
#resize_width = 320
resize_height = int(450)
resize_width = int(800)
dim = [int(resize_height), int(resize_width)]
output_channel= 3

#for iter in range(1,21):   #TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
for iter in range(1):   #TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after each 1000 iters by default.
    saved_state_dict = torch.load(os.path.join('./data/snapshots/g2c1122_93000.pth'))
    if counter==0:
        print snapShot
    counter+=1
    model.load_state_dict(saved_state_dict)

    hist = np.zeros((max_label+1, max_label+1))
    pytorch_list = [];
    #gt = np.zeros((dim[0],dim[1],1,1))

    for i in img_list:
        #img = np.zeros((resize_height,resize_width,3));
        img_original = cv2.imread(os.path.join(im_path,i[:-1]+'.png'))
        gt_temp = np.array(Image.open(os.path.join(gt_path,i[:-1]+'.png')))
        #if (gt_temp.shape[0] < img_original.shape[0]):
            #img_original = cv2.resize(img_original,(gt_temp.shape[1], gt_temp.shape[0]),interpolation=cv2.INTER_NEAREST)
        #elif (gt_temp.shape[0] > img_original.shape[0]):
            #gt_temp = cv2.resize(gt_temp,(img_original.shape[1], img_original.shape[0]),interpolation=cv2.INTER_NEAREST)
        
        #scale_crop_height = random.randint(0, img_original.shape[0]-resize_height)
        #scale_crop_width = random.randint(0, img_original.shape[1]-resize_width)

        new_weidth = int((img_original.shape[1]-(img_original.shape[0]*16//9))//2)
        if new_weidth != 0:
            img_original = img_original[:,new_weidth:-new_weidth]
        img_original = cv2.resize(img_original,(resize_width,resize_height))

        #img_original  = img_original[scale_crop_height:scale_crop_height+resize_height, scale_crop_width:scale_crop_width+resize_width]
        img_temp=img_original.copy()
        img_temp=img_temp.astype(float)

        img_temp[:,:,0] = img_temp[:,:,0] - 104.008  # B
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669  # G
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675  # R

        if new_weidth != 0:
            gt_temp = gt_temp[:,new_weidth:-new_weidth]
        #gt[gt==255] = 0
        gt_temp = cv2.resize(gt_temp, (resize_width, resize_height), interpolation = cv2.INTER_NEAREST)
        
        #gt_temp = gt_temp[scale_crop_height:scale_crop_height+resize_height, scale_crop_width:scale_crop_width+resize_width]
        #gt[:,:,0,0] = gt_temp
        #a = outS(dim)#41
        #b = outS([dim[0]*0.5+1, dim[1]*0.5+1])#21
        #labels = [resize_label_batch(gt,i) for i in [a,a,b,a]]
        #gt_show = labels[output_channel][:,:,0,0]
    
        img_vb = Variable(torch.from_numpy(img_temp[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0)
        output = model(img_vb)
        interp = nn.UpsamplingBilinear2d(size=(resize_height, resize_width))
        output = interp(output[output_channel])
        output = torch.max(output,1)[1]
        output = output[0].cpu().data.numpy()
        #output = output[:,:img_temp.shape[0],:img_temp.shape[1]]

        #output = output.transpose(1,2,0)
        #output = np.argmax(output,axis = 2)
        if args['--visualize']:
            plt.subplot(3, 1, 1)
            plt.imshow(img_original[:,:,::-1])
            plt.subplot(3, 1, 2)
            plt.imshow(gt_temp, vmin = 0, vmax = 34)
            plt.subplot(3, 1, 3)
            plt.imshow(output, vmin = 0, vmax = 34)
            plt.show()
        
        img=Image.fromarray(output.astype(np.uint8))
        img.save("./data/B_semantic/"+i[:-1]+".png")

        #iou_pytorch = get_iou(output,gt)
        #pytorch_list.append(iou_pytorch)
        #hist_ = fast_hist(gt.flatten(),output.flatten(),max_label+1)
        #hist += hist_
        #print iou_pytorch, hist
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print 'pytorch',iter,"Mean iou = ",np.sum(miou)/len(pytorch_list)
