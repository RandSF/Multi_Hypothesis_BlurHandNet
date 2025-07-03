import copy
import math
import torch
import torch.nn as nn

from models.modules.unfolder2d import Unfolder
from models.modules.resnetbackbone import ResNetBackbone
from models.modules.transformer import Transformer
from models.modules.regressor_sim import Regressor
from models.modules.layer_utils import init_weights
from utils.MANO import mano


class BaseModel(nn.Module):
    def __init__(self, opt, weight_init=True):
        super().__init__()
        # define trainable module
        opt_net = opt['network']
        self.backbone = ResNetBackbone()  # backbone
        self.unfolder = Unfolder(opt['task_parameters'])  #  Unfolder
        # self.ktformer = Transformer(opt['task_parameters'])  # KTFormer
        # self.reward_model = RewardModule(**opt_net['reward'], **opt['task_parameters'])
        self.ktformer = Transformer(**opt_net['transformer'], **opt['task_parameters'])
        self.regressor = Regressor(opt['task_parameters'])  # Regressor
        
        # weight initialization
        if weight_init:
            self.backbone.init_weights()
            self.unfolder.apply(init_weights)
            self.ktformer.apply(init_weights)
            self.regressor.apply(init_weights)
            # self.reward_model.apply(init_weights) 
        
        # for producing 3d hand meshs
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()

        self.opt_params = opt['task_parameters']
        
    def forward(self, inputs, targets, meta_info, mode):
        feat_blur, feat_pyramid = self.backbone(inputs['img'])
        
        # extract temporal information via Unfolder
        feat_joint, joint_img = self.unfolder(feat_blur, feat_pyramid)
        
        # feature enhancing via KTFormer
        feat_refine = self.ktformer(feat_joint)
        
        # regress mano shape, pose and camera parameter
        mano_shape, mano_pose, cam_param = self.regressor(feat_blur, feat_refine, joint_img.detach())

        # scores = self.reward_model(mano_pose.detach(), feat_blur, feat_joint)

        # obtain camera translation to project 3D coordinates into 2D space
        # camera translation
        t_xy = cam_param[...,:2]
        gamma = torch.sigmoid(cam_param[...,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.opt_params['focal'][0] * self.opt_params['focal'][1] * self.opt_params['camera_3d_size'] * \
            self.opt_params['camera_3d_size'] / (self.opt_params['input_img_shape'][0] * self.opt_params['input_img_shape'][1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[...,None]), -1)
        
        # obtain 1) projected 3D coordinates 2) camera-centered 3D joint coordinates 3) camera-centered 3D meshes
        ### regress joint and mesh
        data_shape = mano_shape.shape[:-1]
        BS = data_shape.numel()
        root_pose = mano_pose[...,:3].flatten(0, -2)
        hand_pose = mano_pose[...,3:].flatten(0, -2)
        mano_shape = mano_shape.flatten(0, -2)
        cam_trans = cam_trans.flatten(0, -2).unsqueeze(1)

        output = self.mano_layer_right(global_orient=root_pose, hand_pose=hand_pose, betas=mano_shape)
        
        # camera-centered 3D coordinate
        mesh = output.vertices
        joint = torch.bmm(torch.from_numpy(mano.joint_regressor).to(mesh.device)[None,...].expand(BS,-1,-1), mesh)
        
        # project 3D coordinates to 2D space
        x = (joint[...,0] + cam_trans[...,0]) / (joint[...,2] + cam_trans[...,2] + 1e-4) * \
            self.opt_params['focal'][0] + self.opt_params['princpt'][0]
        y = (joint[...,1] + cam_trans[...,1]) / (joint[...,2] + cam_trans[...,2] + 1e-4) * \
            self.opt_params['focal'][1] + self.opt_params['princpt'][1]
        x = x / self.opt_params['input_img_shape'][1]# * self.opt_params['output_hm_shape'][1]
        y = y / self.opt_params['input_img_shape'][0]# * self.opt_params['output_hm_shape'][0]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint[:,mano.root_joint_idx,None,:]
        joint = joint - root_cam

        # add camera translation for the rendering
        mesh = mesh + cam_trans

        J, V = 21, 778
        joint_proj = joint_proj.reshape(*data_shape, J, 2)
        joint = joint.reshape(*data_shape, J, 3)
        mesh = mesh.reshape(*data_shape, V, 3)
        mano_shape = mano_shape.reshape(*data_shape, 10)

        return {
            'pose': mano_pose, 
            'shape': mano_shape,
            'joint_hm': joint_img,
            'joint_proj': joint_proj, 
            'joint': joint, # joint positions in camera coordinates
            'mesh': mesh, 

            'feats': feat_refine,
            # 'score': scores,
            'feats_backbone': {
                'resnet': feat_blur, 
                'unfolder': feat_joint,
            }
        }