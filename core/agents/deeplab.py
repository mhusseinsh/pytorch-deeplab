#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Do 23 Nov 2017 17:37:38 CET
Info:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helpers import read_file, outS, chunker, scale_im, flip, adjust_learning_rate
from core.agent import Agent

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


class DeeplabAgent(Agent):
    def __init__(self, args, model_prototype):
        super(DeeplabAgent, self).__init__(args, model_prototype)
        self.model = self.model_prototype(self.model_params)

        if self.model_file:
            saved_state_dict = torch.load(self.model_file)
            if self.mode==1:
                for i in saved_state_dict:
                    #Scale.layer5.conv2d_list.3.weight
                    i_parts = i.split('.')
                    if i_parts[1]=='layer5':
                        if saved_state_dict[i].size(0)!=self.model.state_dict()[i].size(0):
                            saved_state_dict[i] = self.model.state_dict()[i]
            self.model.load_state_dict(saved_state_dict)

        if self.use_cuda:
            self.model.type(self.dtype)
        self.iter_size = args.iter_size
        self.list_path = args.list_path
        self.img_path = args.img_path
        self.gt_path = args.gt_path
        self.resize_width  = args.resize_width
        self.resize_height = args.resize_height
        self.scale_range   = args.scale_range
        
        self.flip_flag     = args.flip_flag

        self.img_extend_name    = args.img_extend_name
        self.gt_extend_name     = args.gt_extend_name
        if self.mode==1:
            self.optim = args.optim([{'params': get_1x_lr_params_NOscale(self.model), 'lr': self.lr }, {'params': get_10x_lr_params(self.model), 'lr': 10*self.lr} ], lr = self.lr, momentum = 0.9,weight_decay = self.weight_decay)

        if self.mode==2:
            self.with_gt = args.with_gt

    def resize_label_batch(self, label, size, volatile=False):
        label_resized = np.zeros((size[0],size[1],1,label.shape[3]))
        interp = nn.Upsample(size=(size[0], size[1]), mode='bilinear')
        labelVar = Variable(torch.from_numpy(label).type(self.dtype),
                volatile=volatile)
        label_resized = interp(labelVar)
        return label_resized.long()

    def get_train_list(self):
        img_list = read_file(self.list_path)
        self.data_len=len(img_list)
        data_list = []
        for i in range(self.epochs):
            if self.mode == 1:
                np.random.shuffle(img_list)
            data_list.extend(img_list)
        return data_list
    
    def get_data_from_chunk(self, chunk, volatile=False):
        scale = random.uniform(self.scale_range[0], 
                self.scale_range[1])
        dim = [int(scale*self.resize_height), 
                int(scale*self.resize_width)]
        images = []
        gts = []
        for i, piece in enumerate(chunk):
            if self.flip_flag:
                flip_p = random.uniform(0, 1)
            img = cv2.imread(
                    os.path.join(self.img_path, 
                        piece+self.img_extend_name))
            if self.mode==1 or self.with_gt:
                gt = np.array(Image.open(
                    os.path.join(self.gt_path, 
                        piece+self.gt_extend_name))) 
                
                if (gt.shape[0] < img.shape[0]):
                    img = cv2.resize(img,
                            (gt.shape[1], gt.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
                elif (gt.shape[0] > img.shape[0]):
                    gt = cv2.resize(gt,
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
                assert (gt.shape[0] >= self.resize_height)

            crop_height = random.randint(0, img.shape[0]-self.resize_height)
            crop_width = random.randint(0, img.shape[1]-self.resize_width)
            
            img = img[crop_height:crop_height+self.resize_height, 
                    crop_width:crop_width+self.resize_width]
            #img = scale_im(img, scale).astype(np.float32)
            img = scale_im(img, scale).astype(float)
            if self.mode==1 or self.with_gt:
                gt = gt[crop_height:crop_height+self.resize_height,
                        crop_width:crop_width+self.resize_width]
                gt = scale_im(gt, scale, cv2.INTER_NEAREST)
            
            img[:,:,0] = img[:,:,0] - 104.008
            img[:,:,1] = img[:,:,1] - 116.669
            img[:,:,2] = img[:,:,2] - 122.675
            if self.flip_flag: 
                img = flip(img, flip_p)
                gt = flip(gt, flip_p)
            
            images.append(img[np.newaxis, :])
            if self.mode==1 or self.with_gt:
                gts.append(gt[np.newaxis, np.newaxis, :])
        images = np.concatenate(images).transpose(0,3,1,2) 
        if self.mode==1 or self.with_gt:
            gts = np.concatenate(gts) 
            a = outS(dim)
            b = outS([dim[0]*0.5+1, dim[1]*0.5+1])
            gts_vb_list = [self.resize_label_batch(gts,i, volatile) for i in [a,a,b,a]]
        imgs_vb = Variable(torch.from_numpy(images).float().type(self.dtype), volatile=volatile)

        #print(gts_vb_list[0].size(), 
                #gts_vb_list[1].size(),
                #gts_vb_list[2].size(),
                #gts_vb_list[3].size(),
                #type(gts_vb_list[0].data),
                #imgs_vb.size(), 
                #type(imgs_vb.data))
        if self.mode==1 or self.with_gt:
            return imgs_vb, gts_vb_list
        else: 
            return imgs_vb, None

    def fit_model(self):
        self.logger.warning("<===================================> Training ...")
        self.training = True
        self._reset_training_loggings()

        self.start_time = time.time()
        self.step = 0
        self.epoch = 0
        data_gen = chunker(self.get_train_list(), self.batch_size)
        self.max_iter = int(self.epochs * self.data_len//self.batch_size)
        for self.step in range(self.max_iter):
            chunk = data_gen.next()
            imgs_vb, gts_vb_list = self.get_data_from_chunk(chunk)
            out_vb_list = self.model(imgs_vb)
            loss = 0
            for i in range(len(out_vb_list)):
                #print (out_vb_list[i].size())
                loss += F.nll_loss(F.log_softmax(out_vb_list[i]), gts_vb_list[i][0])
            loss = loss/self.iter_size
            loss.backward()
            
            self.logger.warning("Iteration: {}; loss: {}".format(self.step, loss.data.cpu().numpy()*self.iter_size))
            if self.step % self.iter_size == 0:
                self.optim.step()
                self.optim.zero_grad()
                self.lr_poly() 
                adjust_learning_rate(self.optim, self.lr)
                self.logger.warning("Learning_rate: {}".format(self.lr))

            if self.visualize:
                self.writer.add_scalar(self.refs+'/loss', self.iter_size*loss.data[0], self.step)
                
                self.writer.add_image(self.refs+'/RAW', torch.cat((imgs_vb[0,0:1,:,:]+104.008,imgs_vb[0,1:2,:,:]+116.669, imgs_vb[0,2:,:,:]+122.675),dim=0).type(torch.LongTensor), self.step)
                
                self.writer.add_image(self.refs+'/GT', gts_vb_list[3][0]*7, self.step)
                out_img = torch.max(out_vb_list[3], 1)[1]
                self.writer.add_image(self.refs+'/OUT',out_img[0]*7, self.step)
            if self.step % 2000==0:
                self._save_model(self.step)


        self.writer.export_scalars_to_json(os.path.join(self.root_dir, "logs/"+self.refs+'.json'))
        self.writer.close()
    def lr_poly(self, power=0.9):
        multi=((1-float(self.step)/self.max_iter)**(power))
        self.lr = self.lr * multi
        self.weight_decay = self.weight_decay * multi

    def _reset_training_loggings(self):
        pass

    def test_model(self):
        self.logger.warning("<===================================> Testing ...")
        self._load_model(self.model_name)
        data_gen = chunker(self.get_train_list())
        self.max_iter = int(self.data_len)
        self.step = 0
        interp = nn.UpsamplingBilinear2d(size=(self.resize_height, self.resize_width))
        for self.step in range(self.max_iter):
            chunk = data_gen.next()
            imgs_vb, gts_vb_list = self.get_data_from_chunk(chunk, volatile=True)
            print (imgs_vb[0])
            raw_input()
            out_vb_list = self.model(imgs_vb)
            
            output = interp(out_vb_list[3])
            out_img = torch.max(output,1)[1][0]
             
            
            if self.visualize:
                self.writer.add_image(self.refs+'/RAW', torch.cat((imgs_vb[0,0:1,:,:]+104.008,imgs_vb[0,1:2,:,:]+116.669, imgs_vb[0,2:,:,:]+122.675),dim=0).type(torch.LongTensor), self.step)
                if self.with_gt: 
                    self.writer.add_image(self.refs+'/GT', gts_vb_list[3][0]*7, self.step)
                out_img = torch.max(out_vb_list[3], 1)[1]
                self.writer.add_image(self.refs+'/OUT',out_img[0]*7, self.step)
        self.writer.close()
