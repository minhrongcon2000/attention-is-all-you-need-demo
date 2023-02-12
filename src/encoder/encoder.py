import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.mh_attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_head: int,
                 dim_ffn: int=2048,
                 p_drop: float=0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dim_ffn = dim_ffn
        self.p_drop = p_drop
        self.attn = MultiHeadAttention(embed_dim, num_head)
        self.layer_norm_attn = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim_ffn),
            nn.ReLU(),
            nn.Linear(self.dim_ffn, self.embed_dim),
            nn.ReLU(),
        )
        self.layer_norm_ffn = nn.LayerNorm(self.embed_dim)
        
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor=None):
        attn_score = self.attn(inputs, inputs, inputs, mask=mask)
        attn_score = F.dropout(attn_score, self.p_drop)
        attn_score_res = self.layer_norm_attn(inputs + attn_score)
        
        proj = self.ffn(attn_score_res)
        proj = F.dropout(proj, p=self.p_drop)
        return self.layer_norm_ffn(attn_score_res + proj)
        