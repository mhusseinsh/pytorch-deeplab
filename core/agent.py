from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.optim as optim

class Agent(object):
    def __init__(self, args, model_prototype):
        self.mode = args.mode
        self.save_imgs = args.save_imgs
        if self.mode == 2 and self.save_imgs:
            try:
                import scipy.misc
                self.imsave = scipy.misc.imsave
            except ImportError as e: self.logger.warning("WARNING: scipy.misc not found")
            self.img_dir = args.root_dir + "/imgs/" + args.refs + "/"
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
        # logging
        self.logger = args.logger

        self.refs = args.refs
        self.root_dir = args.root_dir

        # prototypes for env & model & loss_model
        self.model_prototype = model_prototype
        self.model_params = args.model_params

        # params
        self.model_name = args.model_name           # NOTE: will save the current model to model_name
        self.model_file = args.model_file           # NOTE: will load pretrained model_file if not None

        self.render = args.render
        self.visualize = args.visualize
        if self.visualize:
            self.vis = args.vis
            self.writer = args.writer
        self.save_best = args.save_best
        if self.save_best:
            self.best_step   = None                 # NOTE: achieves best_reward at this step
            self.best_reward = None                 # NOTE: only save a new model if achieves higher reward

        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # agent_params
        self.criteria = args.criteria
        self.optim = args.optim
        
        # hyperparameters
        self.steps = args.steps
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.test_steps = args.test_steps

    def _load_model(self, model_file):
        if model_file:
            self.logger.warning("Loading Model: " + self.model_file + " ...")
            self.model.load_state_dict(torch.load(model_file))
            self.logger.warning("Loaded  Model: " + self.model_file + " ...")
        else:
            self.logger.warning("No Pretrained Model. Will Train From Scratch.")

    def _save_model(self, step, curr_reward=0.0):
        self.logger.warning("Saving Model    @ Step: " + str(step) + ": " + self.model_name + " ...")
        if self.save_best:
            if self.best_step is None:
                self.best_step   = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step   = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ". {Best Step: " + str(self.best_step) + " | Best Reward: " + str(self.best_reward) + "}")
        else:
            torch.save(self.model.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ".")

    #def _save_model_prog(self, epoch):
        #model_name  = self.root_dir + "/models/" + self.refs + "_" + str(epoch).zfill(3) + ".pth"
        #self.logger.warning("Saving Model    @Epoch: " + str(epoch) + ": " + model_name + " ...")
        #torch.save(self.model.state_dict(), model_name)
        #self.logger.warning("Saving Model    @Epoch: " + str(epoch) + ": " + model_name + ".")

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base calss")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base calss")

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError("not implemented in base calss")

    def fit_model(self):    # training
        raise NotImplementedError("not implemented in base calss")

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError("not implemented in base calss")
