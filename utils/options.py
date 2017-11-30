from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import visdom
import torch
import torch.nn as nn
import torch.optim as optim
from utils.helpers import loggerConfig
from tensorboardX import SummaryWriter

CONFIGS = [
# agent_type, model_type
[ "Deeplab", "Deeplab"   ], # 0
]

class Params(object):   # NOTE: shared across all modules
    def __init__(self):
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)

        # training signature
        self.machine     = "pearl9"    # "machine_id"
        self.timestamp   = "17113001"   # "yymmdd## "
        # training configuration
        self.mode        = 1            # 1(train) | 2(test) | 3(generate image)
        self.config      = 0
        self.train_target   = "semantic" # depth|semantic
        
        self.seed        = 1
        self.render      = False        # whether render the window from the original envs or not
        self.visualize   = True         # whether do online plotting and stuff or not
        self.save_best   = False        # save model w/ highest reward if True, otherwise always save the latest model

        self.agent_type, self.model_type = CONFIGS[self.config]

        self.use_cuda    = torch.cuda.is_available()
        self.dtype       = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # prefix for model/log/visdom
        self.refs        = self.machine + "_" + self.timestamp # NOTE: using this as env for visdom
        self.root_dir    = os.getcwd()

        # model files
        # NOTE: will save the current model to model_name
        self.model_name  = self.root_dir + "/models/" + self.refs + ".pth"
        # NOTE: will load pretrained model_file if not None
        self.model_file  = self.root_dir + "/models/pretrained.pth"
        #self.model_file  = None
        if self.mode == 2:
            self.model_file  = self.model_name
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.refs = self.refs + "_test"    
        if self.mode ==3:
            self.model_file  = self.model_name
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.refs = self.refs + "_generate" 


        # logging configs
        self.log_name    = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger      = loggerConfig(self.log_name, self.verbose)
        self.logger.warning("<===================================>")

        if self.visualize:
            self.vis = visdom.Visdom()
            #self.logger.warning("bash$: python -m visdom.server")           # activate visdom server on bash
            #self.logger.warning("http://localhost:8097/env/" + self.refs)   # open this address on browser
            self.writer = SummaryWriter(self.root_dir + '/runs/' + self.refs)
            self.logger.warning("bash$: tensorboardX --logdir run/")           # activate visdom server on bash
            self.logger.warning("http://localhost:6006")   # open this address on browser

class AgentParams(Params):  # settings for network architecture
    def __init__(self):
        super(AgentParams, self).__init__()

        if self.agent_type == "Deeplab":
            self.optim          = optim.SGD
            self.criteria       = nn.NLLLoss2d(ignore_index=255)
            self.save_imgs      = False
            
            self.steps          = 5000 
            self.batch_size     = 1
            self.lr             = 0.0002
            self.lr_decay       = False
            self.weight_decay   = 0.0005
            self.epochs         = 100
            self.test_steps     = 50
            self.save_freq      = 2000
            
            self.img_path     = self.root_dir+"/data/indoor_img/"
            self.gt_path      = self.root_dir+"/data/indoor_gt/"
            self.img_extend_name    = '.jpg'
            self.gt_extend_name     = '.png'
            self.output_c         = 3  # output which one in the 4 outputs
            self.segmentation_labels = 40
           
            if self.train_target == "depth":
                self.criteria = nn.MSELoss()
                self.gt_path = self.root_dir+"/data/depth/"
                self.gt_extend_name = '.npy'
                # output which one in the 4 outputs
                self.output_c         = 0  
                self.segmentation_labels = 1
                self.lr             = 0.001
            
            self.data_list_file = "indoor_segment_new.txt"
            self.lrflip_flag        = True
            self.udflip_flag        = False    
            self.rotate_flag        = False
            self.crop_width       = 425
            self.crop_height      = 425
            self.scale_range      = [0.7, 0.9]
            self.iter_size        = 8
            if self.mode==2:
                self.data_list_file  = "val.txt"
                self.crop_width       = 1600
                self.crop_height      = 900
                self.batch_size       = 1
                self.epochs           = 1
                self.flip_flag        = False
                self.rotate_flag      = False
                self.scale_range      = [0.2, 0.2]
            elif self.mode==3:
                self.crop_width       = 800
                self.crop_height      = 600
                self.resize_width     = 800
                self.resize_height    = 600
                self.batch_size = 1
                self.epochs = 1
                self.data_list_file  = "img_train03_B.txt"
                self.img_path     = self.root_dir+"/data/train03_B"
                self.flip_flag = False
                self.save_imgs=True
            self.list_path    = self.root_dir+"/data/list/"+self.data_list_file

        self.model_params       = self.segmentation_labels

class Options(Params):
    agent_params  = AgentParams()
