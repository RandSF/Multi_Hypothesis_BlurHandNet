import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops.layers.torch import Rearrange
import os
import math

from .transformer_layers import DecoderBlock, positionalencoding2d, get_timestep_embedding

class Block(nn.Module):
    def __init__(self, embed_dim,
                 num_heads, dropout, mlp_ratio):
        super().__init__()
        # self.joint_ch = dim
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first = True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first = True)
        
        # MLP
        self.linear1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio))
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(int(embed_dim*mlp_ratio), embed_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor + pos
    
    def forward(self, tgt, memory,mask= None,mask_ctx = None,pos_tgt=None, pos_ctx=None):
        B, E = memory.shape[0], memory.shape[-1]
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos_tgt)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt2, pos_tgt),
                                   key=self.with_pos_embed(memory, pos_ctx),
                                   value=memory)[0]
        tgt2 = tgt2.contiguous().view(tgt.shape[0],tgt.shape[1],tgt.shape[2])
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class ScoreNet(nn.Module):
    def __init__(self, 
                 num_k = 16, 
                num_k_select = 4,
                num_timestep = 3, 
                num_joints = 21, 
                embed_dim = 512,
                patch_size = 8,
                input_embed = 512,
                in_chans = 2048,
                num_blocks = 8,
                num_heads = 4, 
                drop_rate = 0.1, 
                mlp_ratio = 4.0,
                alpha = 0.5, **kwargs) -> None:
        super().__init__()
        self.num_k = num_k
        # # pose layer
        self.pose_layer = nn.Sequential(
            nn.Linear(48, num_joints*embed_dim//2),
            Rearrange('k b t je -> (k b t) je'),
            nn.GroupNorm(32, num_channels=num_joints*embed_dim//2),
            Rearrange('(k b t) (j e) -> b k t j e', k=num_k, t=num_timestep, j=num_joints, e=embed_dim//2),
            nn.SiLU(), #nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_rate)
        )
        # feat_joint_layer
        self.joint_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            Rearrange('b t j e -> (b t j) e'),
            nn.GroupNorm(32, num_channels=embed_dim//2),
            Rearrange('(b t j) e -> b t j e', t=num_timestep, j=num_joints),
            nn.SiLU(), #nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_rate)
        )
        # ctx layer
        self.ctx_layer = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1),
            nn.GroupNorm(32, num_channels=embed_dim),
            Rearrange('b c h w -> b (h w) c'),
            nn.SiLU(), #nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_rate)
        )

        # positional embeddings
        pe_ctx = positionalencoding2d(embed_dim, patch_size, patch_size)    # [E, H, W]
        # pe_t = get_timestep_embedding(torch.arange(num_timestep), embed_dim)    # [T, E]
        self.register_buffer('pe_ctx', None, persistent=False)   # [1, HW, E]
        self.register_buffer('pe_t', None, persistent=False)   # [1, 1, T, 1, E]
        
        self.pe_ctx = pe_ctx.flatten(-2).transpose(-1, -2).unsqueeze(0)
        # self.pe_t = pe_t[None, None, :, None, :]
        self.pe_t = nn.Parameter(torch.zeros([1, 1, num_timestep, 1, embed_dim]))
        self.pe_j = nn.Parameter(torch.zeros([1, 1, 1, num_joints, embed_dim]))
        self.pe_k = nn.Parameter(torch.zeros([1, num_k, 1, 1, embed_dim]))

        trunc_normal_(self.pe_t, std=.02)
        trunc_normal_(self.pe_j, std=.02)
        trunc_normal_(self.pe_k, std=.02)
        
        # blocks
        dpr = [x.item() for x in torch.linspace(0, drop_rate, num_blocks)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, dpr[i], int(mlp_ratio))
        for i in range(num_blocks)] )
        # score head
        self.last_norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Sequential(
            Rearrange("b (k t j) e -> (b k) e t j", k=num_k, t=num_timestep, j=num_joints),
            nn.Conv2d(embed_dim, embed_dim, (num_timestep, 1)), # mix timestep channel
            nn.SiLU(), #nn.ReLU(),
            nn.Dropout(p=drop_rate),
            Rearrange('(b k) e t j -> b k t (j e)', k=num_k),
            nn.Linear(num_joints*embed_dim, 1024),
            nn.SiLU(), #nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024,2)   # for JRC
        )
        


    def forward(self, pose, joint_cam, ctx, feat_joint):
        '''
        ctx: [B, C, H, W]
        feat_mano: [B, K, T, J, E]
        pose: [B, K, T, theta]
        joint_cam: [B, K, T, J, 3]

        '''
        B, T, J, E = feat_joint.shape
        K = self.num_k

        xin_pose = self.pose_layer(pose)
        xin_joint = self.joint_layer(feat_joint).unsqueeze(1).expand_as(xin_pose)   # [B, K, T, J, E/2]
        x = torch.cat([xin_pose, xin_joint], dim=-1) # [B, K, T, J, E]

        pe_x = (self.pe_t + self.pe_j + self.pe_k).expand_as(x) # [B, K, T, J, E]
        pe_ctx = self.pe_ctx
        ctx = self.ctx_layer(ctx)
        
        # pe_ctx = self.pe_ctx    # [1, HW, E]
        # x = x.reshape(B, K*T*J, E)
        # ctx = xin_joint.reshape(B, T*J, E)
        # ctx = ctx.repeat(K, 1, 1)
        # for block in self.blocks:
        #     x = block(tgt=x, memory=ctx,
        #               pos_tgt=pe_x.reshape(B*K, T*J, -1), 
        #               pos_ctx=pe_x.reshape(B*K, T*J, -1))

        # x = x.reshape(B, K, T, J*E)
        x = x.reshape(B, K*T*J, E)+pe_x.reshape(B, K*T*J, -1)
        ctx = ctx + pe_ctx
        for block in self.blocks:
            x = block(tgt=x, mem=ctx)

        score_feature = self.last_norm(x)

        score = self.score_head(score_feature)# [B, K, T, 2]

        # print("feat_mano", torch.argwhere(torch.isnan(feat_mano)).shape)
        # print("xin_mano", torch.argwhere(torch.isnan(xin_mano)).shape)
        # print("xin_joint", torch.argwhere(torch.isnan(xin_joint)).shape)
        # print("ctx", torch.argwhere(torch.isnan(ctx)).shape)
        # print("x", torch.argwhere(torch.isnan(x)).shape)
        # print("score_feature", torch.argwhere(torch.isnan(score_feature)).shape)

        # return score.reshape(B, K, T, 2).transpose(0, 1)
        return score.reshape(B, K, 2).transpose(0, 1)