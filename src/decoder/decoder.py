import sys
sys.path.append("..")

import torch.nn as nn
import torch.nn.functional as F

from attention.mh_attention import MultiHeadAttention


class Decoder(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_head: int,
                 dim_ffn: int=2048,
                 p_drop: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dim_ffn = dim_ffn
        self.p_drop = p_drop
        
        self.attn = MultiHeadAttention(self.embed_dim, self.num_head)
        self.layer_norm_self_attn = nn.LayerNorm(self.embed_dim)
        
        self.cross_attn = MultiHeadAttention(self.embed_dim, self.num_head)
        self.layer_norm_cross_attn = nn.LayerNorm(self.embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim_ffn),
            nn.ReLU(),
            nn.Linear(self.dim_ffn, self.embed_dim),
            nn.ReLU(),
        )
        self.layer_norm_ffn = nn.LayerNorm(self.embed_dim)
        
    def forward(self, src_embed, tgt_embed):
        self_attn_score = self.attn(tgt_embed, tgt_embed, tgt_embed, mask=True)
        self_attn_score = F.dropout(self_attn_score, p=self.p_drop)
        self_attn_score_res = self.layer_norm_self_attn(tgt_embed + self_attn_score)
        
        cross_attn_score = self.cross_attn(src_embed, self_attn_score_res, src_embed)
        cross_attn_score = F.dropout(cross_attn_score, p=self.p_drop)
        cross_attn_score_res = self.layer_norm_cross_attn(self_attn_score_res + cross_attn_score)
        
        proj = self.ffn(cross_attn_score_res)
        proj = F.dropout(proj, p=self.p_drop)
        return self.layer_norm_ffn(cross_attn_score_res + proj)
        
