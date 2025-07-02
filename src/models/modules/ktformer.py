import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

class KTFormer(nn.Module):
    def __init__(self, opt_params, in_chans=512, embed_dim=512,
                 num_blocks=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_joints = opt_params['num_joints']

        self.patch_embed = nn.Linear(in_chans, embed_dim)
        self.pos_embed_t = nn.Parameter(torch.randn(1, 3, 1, embed_dim))  # time direction
        self.pos_embed_j = nn.Parameter(torch.randn(1, 1, self.num_joints, embed_dim))  # joint direction
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_blocks)])

    def forward(self, feat_joint):
        # concat the joint features
        # x = torch.cat((feat_joint_e1, feat_joint_md, feat_joint_e2), dim=1)

        
        # forwarding transformer block
        B, T, J, E = feat_joint.shape
        x = self.patch_embed(feat_joint)
        x = (x + self.pos_embed_t + self.pos_embed_j).flatten(1, 2)
        for blk in self.blocks:
            x = blk(x)
            
        # channel-wise dividing operation
        feat_joint = x
        
        return feat_joint.reshape(B, T, J, -1)

class TransformerFinger(nn.Module):
    def __init__(self, opt_params, in_chans=512, embed_dim=512,
                 num_blocks=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_joints = opt_params['num_joints']

        self.patch_embed = nn.Linear(in_chans, embed_dim)
        self.pos_embed_t = nn.Parameter(torch.randn(1, 3, 1, embed_dim))  # time direction
        self.pos_embed_j = nn.Parameter(torch.randn(1, 1, self.num_joints, embed_dim))  # joint direction
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_blocks)])

        self.token_finger = nn.Parameter(torch.zeros(1, 3, 6, embed_dim))
        from models_current.modules.transformer_layers import EncoderBlock
        self.finger_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads)
            for _ in range(num_blocks)])
        self.register_buffer('mask', None, persistent=False)
        self.mask = _build_mask()

        from timm.layers import trunc_normal_
        trunc_normal_(self.pos_embed_t)
        trunc_normal_(self.pos_embed_j)
        trunc_normal_(self.token_finger)

    def forward(self, feat_joint):
        # concat the joint features
        # x = torch.cat((feat_joint_e1, feat_joint_md, feat_joint_e2), dim=1)

        
        # forwarding transformer block
        B, T, J, E = feat_joint.shape
        x = self.patch_embed(feat_joint)
        x = (x + self.pos_embed_t + self.pos_embed_j).flatten(1, 2)
        for blk in self.blocks:
            x = blk(x)
            
        # channel-wise dividing operation
        x = x.reshape(B, T, J, -1)
        x = torch.cat([x, self.token_finger.expand(B,-1,-1,-1)], dim=-2).flatten(1, 2)

        for block in self.finger_blocks:
            x = block(x, self.mask)
        
        feat_joint = x.reshape(B, T, J+6, -1)[:,:,-6:]
        return feat_joint

def _build_mask():
    T, J, F = 3, 21, 6
    mask = torch.zeros(J+F, J+F)
    mask[:J, :J] = 1
    mask[J,:] = 1   # root can access all tokens
    # thumb
    mask[J+5, 0] = mask[J+5, 1] = mask[J+5, 2] = mask[J+5, 3] = mask[J+5, 4] = 1
    # index
    mask[J+1, 0] = mask[J+1, 5] = mask[J+1, 6] = mask[J+1, 7] = mask[J+1, 8] = 1
    # middle
    mask[J+2, 0] = mask[J+2, 9] = mask[J+2,10] = mask[J+2,11] = mask[J+2,12] = 1
    # ring
    mask[J+4, 0] = mask[J+4,13] = mask[J+4,14] = mask[J+4,15] = mask[J+4,16] = 1
    # little
    mask[J+3, 0] = mask[J+3,17] = mask[J+3,18] = mask[J+3,19] = mask[J+3,20] = 1

    return ~mask.repeat(T, T).bool()

