import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops.layers.torch import Rearrange
import os
import math

# from models_cap_new.modules.unfolder_simple import Unfolder
from models.modules.unfolder import Unfolder
from models.modules.transformer_layers import DecoderBlock, get_timestep_embedding, positionalencoding2d

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
                drop_rate = 0.2, 
                mlp_ratio = 4.0,
                alpha = 0.5, **kwargs) -> None:
        super().__init__()
        self.num_k = num_k
        self.num_k_select = num_k_select
        
        # # pose layer
        self.pose_layer = nn.Sequential(
            nn.Linear(48, num_joints*embed_dim//2),
            Rearrange('b k t je -> (b k t) je'),
            nn.GroupNorm(32, num_channels=num_joints*embed_dim//2),
            Rearrange('(b k t) (j e) -> b k t j e', k=num_k, t=num_timestep, j=num_joints, e=embed_dim//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_rate)
        )

        self.joint_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            Rearrange('b t j e -> (b t j) e'),
            nn.GroupNorm(32, num_channels=embed_dim//2),
            Rearrange('(b t j) e -> b t j e', t=num_timestep, j=num_joints),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_rate)
        )
        # ctx layer
        self.ctx_layer = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1),
            nn.GroupNorm(32, num_channels=embed_dim),
            Rearrange('b c h w -> b (h w) c'),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_rate)
        )

        # positional embeddings
        pe_ctx = positionalencoding2d(embed_dim, patch_size, patch_size)    # [E, H, W]
        self.register_buffer('pe_ctx', pe_ctx.flatten(-2).transpose(-1, -2).unsqueeze(0))   # [1, HW, E]
        pe_t = get_timestep_embedding(torch.arange(num_timestep), embed_dim)    # [T, E]
        self.register_buffer('pe_t', pe_t[None, None, :, None, :])   # [1, 1, T, 1, E]
        self.pe_j = nn.Parameter(torch.zeros([1, 1, 1, num_joints, embed_dim]))
        trunc_normal_(self.pe_j, std=.02)
        self.pe_k = nn.Parameter(torch.zeros([1, num_k, 1, 1, embed_dim]))
        trunc_normal_(self.pe_k, std=.02)
        
        # blocks
        dpr = [x.item() for x in torch.linspace(0, drop_rate, num_blocks)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, dpr[i])
        for i in range(num_blocks)] )
        # score head
        self.last_norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Sequential(
            Rearrange("bk (t j) e -> bk e t j", t=num_timestep, j=num_joints),
            nn.Conv2d(embed_dim, embed_dim, (num_timestep, 1)), # mix timestep channel
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            Rearrange('(b k) e t j -> b k (t j e)', k=self.num_k), # t=1
            nn.Linear(num_joints*embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024,2)   # for JRC
        )

    def forward(self, feat_mano, pose, joint_cam, ctx, feat_joint):
        '''
        ctx: [B, C, H, W]
        feat_mano: [B, K, T, J, E]
        pose: [B, K, T, theta]
        joint_cam: [B, K, T, J, 3]

        '''
        B, K, T, J, E = feat_mano.shape
        # K = self.num_k

        ctx = self.ctx_layer(ctx) # [B, HW, E]
        
        # xin_mano = self.mano_layer(feat_mano)   # [B, K, T, J, E/2]
        xin_pose = self.pose_layer(pose)
        xin_joint = self.joint_layer(feat_joint).unsqueeze(1).expand_as(xin_pose)        # [B, K, T, J, E/2]
        x = torch.cat([xin_joint, xin_pose], dim=-1) # [B, K, T, J, E]
        
        pe_x = (self.pe_t + self.pe_j + self.pe_k).expand_as(x) # [B, K, T, J, E]
        # pe_x = (self.pe_t + self.pe_j).expand_as(x) # [B, K, T, J, E]
        
        pe_ctx = self.pe_ctx    # [1, HW, E]
        x = x.reshape(B*K, T*J, E)
        ctx = ctx.repeat(K, 1, 1)
        for block in self.blocks:
            x = block(tgt=x, memory=ctx,
                      pos_tgt=pe_x.reshape(B*K, T*J, -1), 
                      pos_ctx=pe_ctx)

        score_feature = self.last_norm(x)

        score = self.score_head(score_feature)# [BK, 2]

        return score.reshape(B, K, 2)   # [B, K, 2]
    
if __name__ == '__main__':
    
    md = ScoreNet()

    x = dict(img=torch.rand([24, 3, 256, 256]),
             feat_mano=torch.rand([24, 16, 3, 21, 512]), 
             pose=torch.rand([24, 16, 3, 48]), 
             joint_cam=torch.rand([24, 16, 3, 21, 3]), 
             ctx=torch.rand([24, 2048, 8, 8]),)
    
    targ = dict(joint_img=torch.rand([24, 21, 3]), 
                joint_img_past=torch.rand([24, 21, 3]), 
                joint_img_future=torch.rand([24, 21, 3]), )
    meta = dict(joint_trunc=torch.ones([24, 21, 1]), 
                joint_trunc_past=torch.ones([24, 21, 1]), 
                joint_trunc_future=torch.ones([24, 21, 1]), )
    
    opt = torch.optim.Adam()
    

