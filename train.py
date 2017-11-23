import torch
import torch.nn as nn
import numpy as np
import pickle
import deeplab_resnet
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import matplotlib.pyplot as plt
from tqdm import *
import random
from docopt import docopt
import timeit
from PIL import Image
from tensorboardX import SummaryWriter
import torch.nn.functional as F

writer = SummaryWriter()
start = timeit.timeit
docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization

Usage:
    train.py [options]

Options:
    -h, --help                   Print this message
    --GTpath=<str>               Ground truth path prefix [default: data/gt/]
    --IMpath=<str>               Sketch images path prefix [default: data/img/]
    --NoLabels=<int>             The number of different labels in training data, VOC has 21 labels, including background [default: 35]
    --LISTpath=<str>             Input image number list file [default: data/list/train_aug.txt]
    --lr=<float>                 Learning Rate [default: 0.00025]
    -i, --iterSize=<int>         Num iters to accumulate gradients over [default: 8]
    --wtDecay=<float>            Weight decay during training [default: 0.0005]
    --gpu0=<int>                 GPU number [default: 0]
    --maxIter=<int>              Maximum number of iterations [default: 200000]
    --saveSnap=<str>             Maximum number of iterations [default: g2c1122]
"""

#    -b, --batchSize=<int>       num sample per batch [default: 1] currently only batch size of 1 is implemented, arbitrary batch size to be implemented soon
args = docopt(docstr, version='v0.1')
print(args)

cudnn.enabled = False
gpu0 = int(args['--gpu0'])

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

def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size[0],size[1],1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size[0], size[1]))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (int(img_temp.shape[1]*scale),int(img_temp.shape[0]*scale))
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (int(img_temp.shape[1]*scale),int(img_temp.shape[0]*scale))
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)

def get_data_from_chunk_v2(chunk, iter):
    gt_path =  args['--GTpath']
    img_path = args['--IMpath']

    resize_height = 513
    resize_width = 513
    
    scale = random.uniform(0.6, 0.8) #random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch, so we random scaling in the range (0.5,1.3), different than caffe implementation in that caffe used only 4 fixed scales. Refer to read me
    

    dim = [int(scale*resize_height), int(scale*resize_width)]
    images = np.zeros((dim[0],dim[1],3,len(chunk)))
    gt = np.zeros((dim[0],dim[1],1,len(chunk)))

    for i,piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        img_original = cv2.imread(os.path.join(img_path,piece+'.png'))
        gt_temp = np.array(Image.open(os.path.join(gt_path,piece+'.png')))
        if (gt_temp.shape[0] < img_original.shape[0]):
            img_original = cv2.resize(img_original,(gt_temp.shape[1], gt_temp.shape[0]),interpolation=cv2.INTER_NEAREST)
        elif (gt_temp.shape[0] > img_original.shape[0]):
            gt_temp = cv2.resize(gt_temp,(img_original.shape[1], img_original.shape[0]),interpolation=cv2.INTER_NEAREST)
        
        scale_crop_height = random.randint(0, img_original.shape[0]-resize_height)
        scale_crop_width = random.randint(0, img_original.shape[1]-resize_width)
        
        #new_width = int((img_original.shape[1]-(img_original.shape[0]*16//9))//2)
        #img_temp = img_original[:, new_width:-new_width]
        #img_temp = cv2.resize(img_temp,(resize_width,resize_height)).astype(float)
        
        img_original  = img_original[scale_crop_height:scale_crop_height+resize_height, scale_crop_width:scale_crop_width+resize_width]
        img_temp = img_original[:,:,:].astype(float)
        img_temp = scale_im(img_temp,scale)

        if iter%20==0:
            writer.add_image('input', torch.from_numpy(img_temp.transpose(2,0,1)).type(torch.LongTensor), iter)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = flip(img_temp,flip_p)
        images[:,:,:,i] = img_temp

        #gt_temp = np.array(Image.open(os.path.join(gt_path,piece+'.png')))[:, new_width:-new_width]
        #gt_temp[gt_temp == 255] = 0
        #gt_temp = cv2.resize(gt_temp,(resize_width,resize_height),interpolation=cv2.INTER_NEAREST)
        gt_temp = gt_temp[scale_crop_height:scale_crop_height+resize_height, scale_crop_width:scale_crop_width+resize_width]
        gt_temp = scale_gt(gt_temp,scale)

        gt_temp = flip(gt_temp,flip_p)
        gt[:,:,0,i] = gt_temp
        #a = outS(321*scale)#41
        #b = outS((321*0.5)*scale+1)#21
        a = outS(dim)#41
        b = outS([dim[0]*0.5+1, dim[1]*0.5+1])#21
    # output is not as the same as the original image
    labels = [resize_label_batch(gt,i) for i in [a,a,b,a]]
    images = images.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    return images, labels

def loss_calc(out, label, gpu0, iter, count):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)

    #m = nn.LogSoftmax()
    #criterion = nn.NLLLoss2d()
    #out = m(out)
    if iter%20==0 and count==0:
        writer.add_image('GT'+str(count), label[0]*7, iter)
        out_img = torch.max(out.data, 1)[1]
        writer.add_image('OUT'+str(count), out_img[0]*7, iter)

    #return criterion(out,label)
    return F.nll_loss(F.log_softmax(out),label)

def lr_poly(base_lr, base_wd, iter,max_iter,power):
    
    multi=((1-float(iter)/max_iter)**(power))
    return base_lr * multi, base_wd * multi

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

if not os.path.exists('data/snapshots'):
    os.makedirs('data/snapshots')

model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))

saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
if int(args['--NoLabels'])!=21:
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if i_parts[1]=='layer5':
            saved_state_dict[i] = model.state_dict()[i]

model.load_state_dict(saved_state_dict)

max_iter = int(args['--maxIter'])
batch_size = 1
base_weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.eval() # use_global_stats = True

img_list = read_file(args['--LISTpath'])

data_list = []
for i in range(100):  # make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
    np.random.shuffle(img_list)
    data_list.extend(img_list)

model.cuda(gpu0)
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = base_weight_decay)

optimizer.zero_grad()
data_gen = chunker(data_list, batch_size)

for iter in range(max_iter+1):
    chunk = data_gen.next()

    images, label = get_data_from_chunk_v2(chunk, iter)
    images = Variable(images).cuda(gpu0)

    out = model(images)
    loss = loss_calc(out[0], label[0], gpu0, iter, 0)
    iter_size = int(args['--iterSize'])
    for i in range(len(out)-1):
        loss = loss + loss_calc(out[i+1], label[i+1], gpu0, iter, i+1)
    loss = loss/iter_size
    loss.backward()
    if iter%20 == 0:
        writer.add_image('RAW', torch.cat((images[0,0:1,:,:]+104.008,images[0,1:2,:,:]+116.669, images[0,2:,:,:]+122.675),dim=0).type(torch.LongTensor), iter)

    if iter %1 == 0:
        print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())
        writer.add_scalar('loss', iter_size*loss.data[0], iter)
    if iter % iter_size  == 0:
        optimizer.step()
        lr_, weight_decay_ = lr_poly(base_lr, base_weight_decay, iter,max_iter,0.9)
        print '(poly lr policy) learning rate',lr_
        #optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay_)
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = base_weight_decay)
        optimizer.zero_grad()

    if iter % 1000 == 0 and iter!=0:
        print 'taking snapshot ...'
        torch.save(model.state_dict(),'data/snapshots/'+args['--saveSnap']+'_'+str(iter)+'.pth')
end = timeit.timeit
print end-start,'seconds'
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
