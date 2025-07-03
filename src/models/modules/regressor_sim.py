import torch
import torch.nn as nn

from models.modules.layer_utils import make_linear_layers
from utils.MANO import mano
from utils.transforms import rot6d_to_axis_angle

# for applying KTD
ANCESTOR_INDEX = [
    [],  # Wrist
    [0],  # Index_1
    [0,1],  # Index_2
    [0,1,2],  # index_3
    [0],  # Middle_1
    [0,4],  # Middle_2
    [0,4,5],  # Middle_3
    [0],  # Pinky_1
    [0,7],  # Pinky_2
    [0,7,8],  #Pinky_3
    [0],  # Ring_1
    [0,10],  # Ring_2
    [0,10,11],  # Ring_3
    [0],  # Thumb_1
    [0,13],  # Thumb_2
    [0,13,14]  # Thumb_3
]
class Regressor(nn.Module):
    def __init__(self, opt_params, in_chans=2048, in_chans_pose=512):
        super().__init__()
        # mano shape regression, multiply the output channel by 3 to account e1, m, and e2
        self.shape_out = make_linear_layers([in_chans, mano.shape_param_dim * 3], relu_final=False)
        
        # mano pose regression, apply KTD using ancestral pose parameters
        self.joint_regs = nn.ModuleList()
        for ancestor_idx in ANCESTOR_INDEX:
            regressor = nn.Linear(opt_params['num_joints']*(2+in_chans_pose) + 6 * len(ancestor_idx), 6)
            self.joint_regs.append(regressor)
            
        # camera parameter regression for projection loss, multiply 3 to account e1, m, and e2
        self.cam_out = make_linear_layers([in_chans, 3 * 3], relu_final=False)

    def forward(self, feat_blur, feat_joint, joint_img):
        
        # mano pose parameter regression
        K, B, T, J, E = feat_joint.shape
        joint_img = joint_img[None].expand(K, -1, -1, -1, -1)
        feat_one = torch.cat((feat_joint, joint_img), -1).flatten(-2)   # [K, B, T, -1]

        pose_6d_list = []

        # regression using KTD
        for j, (ancestor_idx, reg) in enumerate(zip(ANCESTOR_INDEX, self.joint_regs)):
            ances = torch.cat([feat_one] + [pose_6d_list[i] for i in ancestor_idx], dim=-1)
            pose_6d_list.append(reg(ances))
            
        pose_6d = torch.cat(pose_6d_list, dim=-1)
        
        # change 6d pose -> axis angles
        mano_pose = rot6d_to_axis_angle(pose_6d.reshape(-1, 6)).reshape(K, B, T, mano.orig_joint_num * 3)
        
        # mano shape parameter regression
        shape_param = self.shape_out(feat_blur.mean((2,3))).reshape(B, T, -1)[None].expand(K, -1, -1, -1)
        mano_shape = shape_param
        # camera parameter regression
        cam_param = self.cam_out(feat_blur.mean((2,3))).reshape(B, T, -1)[None].expand(K, -1, -1, -1)

        return mano_shape, mano_pose, cam_param