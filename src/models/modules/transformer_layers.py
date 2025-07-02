import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

from typing import Optional

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb  # [T, E]

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                        -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe   # [E, H, W]

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0, ratio=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        self.linear1 = nn.Linear(dim, dim*ratio)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim*ratio, dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt,
                src_mask: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=src_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt
    
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0, ratio=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        self.linear1 = nn.Linear(dim, dim*ratio)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim*ratio, dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, mem, #mem_pos: Optional[Tensor] = None, 
                src_mask: Optional[Tensor] = None, cross_mask: Optional[Tensor] = None):
        
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=src_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        key = mem# + mem_pos
        value = mem
        tgt2 = self.cross_attn(tgt2, key, value, attn_mask=cross_mask)[0]
        tgt = tgt + self.dropout2(tgt2)


        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    
class ResidualBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(indim, indim//2, 1, 1, 0), 
            nn.BatchNorm2d(indim//2), 
            nn.Conv2d(indim//2, indim//2, 3, 1, 1), 
            nn.BatchNorm2d(indim//2), 
            nn.Conv2d(indim//2, outdim, 1, 1, 0), 
            nn.BatchNorm2d(outdim), 
        )
        self.down_sample = nn.Sequential(nn.Conv2d(indim, outdim, 1, 1, 0), nn.BatchNorm2d(outdim)) if indim!=outdim else nn.Identity()

        self.act = nn.ReLU()

    def forward(self, x):
        
        identity = self.down_sample(x)
        y = self.layer(x)

        return self.act(y + identity)
