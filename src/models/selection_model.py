import copy
import math
import torch
import torch.nn as nn

from models.modules.unfolder2d import Unfolder
from models.modules.resnetbackbone import ResNetBackbone
from models.modules.reward_model import ScoreNet
from models.modules.layer_utils import init_weights
from utils.MANO import mano



class SelectionModel(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()
        # define trainable module
        opt_net = opt['network']
        self.backbone = ResNetBackbone()  # backbone
        self.unfolder = Unfolder(opt['task_parameters'])  #  Unfolder
        self.scorenet = ScoreNet(**opt_net['reward'], **opt['task_parameters'])
        
        # weight initialization
        if weight_init:
            self.backbone.init_weights()
            self.unfolder.apply(init_weights)
            self.scorenet.apply(init_weights)

        self.opt_params = opt['task_parameters']
        
    def forward(self, inputs, targets, meta_info, mode):
        feat_blur, feat_pyramid = self.backbone(inputs['img'])
        
        # extract temporal information via Unfolder
        feat_joint, joint_img = self.unfolder(feat_blur, feat_pyramid)

        # feat_blur = inputs['feats_backbone']['resnet']
        # feat_joint = inputs['feats_backbone']['unfolder']

        pose = inputs['pose'].detach()
        joint = inputs['joint'].detach()
        scores = self.scorenet(pose, joint, feat_blur, feat_joint)
        return {
            'joint_hm_rm': joint_img, 
            'score': scores,
        }