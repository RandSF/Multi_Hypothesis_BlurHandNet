import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .adjoint_matrix import build_temporal_spatial_adjoint_matrix
from .transformer_layers import EncoderBlock, DecoderBlock, ResidualBlock, positionalencoding2d


class Transformer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, num_blocks, num_heads, num_k, num_timestep, num_joints, dropout=0, ratio=4, 
                 **kwargs):
        super().__init__()
        self.dim = embed_dim
        self.num_timestep = num_timestep
        self.num_joints = num_joints
        self.num_k = num_k

        # self.candidate_layer = nn.ModuleList([Mlp(embed_dim) for _ in range(num_k)])
        # self.grid_layer = ResidualBlock(2048, embed_dim)
        # self.in_proj_layer = nn.Linear(in_dim, embed_dim)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout, ratio) for _ in range(num_blocks)
        ])


        # self.out_proj_layer = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))

        self.hypo_embed = nn.Parameter(torch.zeros(1, num_k, 1, 1, embed_dim))  # time direction
        self.pos_embed_t = nn.Parameter(torch.zeros(1, 1, num_timestep, 1, embed_dim))  # time direction
        self.pos_embed_j = nn.Parameter(torch.zeros(1, 1, 1, num_joints, embed_dim))  # joint direction
        # self.query_fingers = nn.Parameter(torch.zeros(1, num_timestep*6, embed_dim)) # finger & root queries

        trunc_normal_(self.hypo_embed, std=.02)
        trunc_normal_(self.pos_embed_t, std=.02)
        trunc_normal_(self.pos_embed_j, std=.02)
        # trunc_normal_(self.query_fingers, std=.02)

        # self.register_buffer("masks", None, persistent=False)
        # mats = build_temporal_spatial_adjoint_matrix(k=num_blocks)   # [k, TJ, TJ]
        # self.masks = ~F.pad(mats, (0,64, 0,64), mode='constant', value=1)
        # self.masks = ~mats
        
    def forward(self, feat_joint):
        # feat_joint: [B, T, J, E]
        K = self.num_k
        B, T, J, _ = feat_joint.shape


        x = (feat_joint.unsqueeze(1) + self.hypo_embed + self.pos_embed_t + self.pos_embed_j).flatten(1, 3)
        for block in self.blocks:
            x = block(x)
        mem = x

        feat_refine = x.reshape(B, K, T, J, -1).transpose(0, 1)

        return feat_refine
