import torch
from torch import Tensor, nn
from torch.nn import functional as F
from timm.layers import trunc_normal_

from .adjoint_matrix import build_adjoint_matrix
from .transformer_layers import EncoderBlock, ResidualBlock, positionalencoding2d
from utils.MANO import mano

mano_joints = mano.orig_joint_num

class RewardModule(nn.Module):
    def __init__(self, ctx_dim, embed_dim, num_heads, num_blocks, 
                 num_timestep, num_joints, dropout=0,
                 **kwargs):
        super().__init__()
        self.dim = embed_dim
        self.num_timestep = num_timestep
        self.num_joints = num_joints
        
        ### add extra embedding layer to fit selection model
        self.ctx_layer = ResidualBlock(ctx_dim, embed_dim)

        self.hypo_layer = nn.Sequential(
            nn.Linear(48, embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, 2*embed_dim), 
            nn.ReLU(),
            nn.Linear(2*embed_dim, embed_dim), 
            nn.ReLU(),
        )

        self.feat_layer = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim), 
            nn.ReLU(),
            nn.Linear(2*embed_dim, embed_dim), 
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

        self.score_head = nn.ModuleList([nn.Linear(embed_dim, 2) for _ in range(num_timestep)])

        self.pos_embed_s = nn.Parameter(torch.zeros(1, num_timestep, embed_dim))
        self.pos_embed_t = nn.Parameter(torch.zeros(1, num_timestep, 1, embed_dim))  # time direction
        self.pos_embed_j = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))  # joint direction
        # self.query_fingers = nn.Parameter(torch.zeros(1, num_timestep*6, embed_dim)) # finger & root queries


        trunc_normal_(self.pos_embed_t, std=.02)
        trunc_normal_(self.pos_embed_j, std=.02)

        # self.register_buffer("tgt_attn_masks", None, persistent=False)
        # mats = build_adjoint_matrix(num_joint=16, k=num_blocks) # [k, J, J]
        # mats = F.pad(mats, (0,1,0,0), mode='constant', value=0)    # mask the attn weight from joints to score token
        # mats = F.pad(mats, (0,0,0,1), mode='constant', value=1)    # score token can access all other tokens
        # self.tgt_attn_masks = ~mats.repeat(1, num_timestep, num_timestep)    # temporally-spatial token

        # self.register_buffer("ctx_attn_masks", None, persistent=False)
        # mats = build_adjoint_matrix(k=num_blocks)
        # self.ctx_attn_masks = ~ mats


    def forward(self, pose ,feat_blur, feat_joint):
        # ctx_grid: [B, C, h, w]
        # pose: [K, B, T, 48]
        # feat_joint: [B, T, J, E], J_=16
        B, C, H, W = feat_blur.shape
        K = pose.shape[0]
        T, J, E = self.num_timestep, self.num_joints, self.dim

        # process ctx
        feat_pose = self.hypo_layer(pose).flatten(0, 1)   # [KB, T, E]

        feat_fuse = self.feat_layer(feat_joint).flatten(1, 2)[None].expand(K, -1, -1, -1).flatten(0, 1)

        feat_grid = self.ctx_layer(feat_blur).flatten(-2).transpose(1, 2)   # [B, hw, E]
        feat_grid = feat_grid[None].expand(K, -1, -1, -1).flatten(0, 1)

        pe_img = positionalencoding2d(E, H, W).to(feat_grid.device).flatten(1).transpose(0, 1)
        pe = torch.cat([self.pos_embed_s, (self.pos_embed_t+self.pos_embed_j).flatten(1, 2), pe_img[None]], dim=1)

        x = torch.cat([feat_pose, feat_fuse, feat_grid], dim=1) + pe
        for block in self.blocks:
            x = block(x)

        score_feat = x[:,:T].reshape(K, B, T, -1)

        score_list = [head(score_feat[:,:,t]) for t, head in enumerate(self.score_head)]

        return torch.stack(score_list, dim=-2)  # [K, B, T, 2]
        
