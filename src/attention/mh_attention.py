import torch
import torch.nn as nn
import torch.nn.functional as F

from .validator import validate_constructor_args, validate_forward_input_args


@validate_constructor_args
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_head: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        
        self.key_dim = embed_dim // num_head
        self.value_dim = embed_dim // num_head
        
        self.keys_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.queries_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.values_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.scores = nn.Linear(self.embed_dim, self.embed_dim)
        
    @validate_forward_input_args
    def forward(self, 
                key: torch.Tensor, 
                query: torch.Tensor, 
                value: torch.Tensor,
                mask: torch.Tensor=None) -> torch.Tensor:
        N, S_in, _ = key.shape
        N, S_out, _ = query.shape
        
        key_embed: torch.Tensor = self.keys_proj(key)\
                                      .reshape(N, S_in, self.num_head, self.key_dim)\
                                      .transpose(1, 2) # (N, H, S, K)
        query_embed: torch.Tensor = self.queries_proj(query)\
                                        .reshape(N, S_out, self.num_head, self.key_dim)\
                                        .transpose(1, 2) # (N, H, S, K)
        value_embed: torch.Tensor = self.values_proj(value)\
                                        .reshape(N, S_in, self.num_head, self.value_dim)\
                                        .transpose(1, 2) # (N, H, S, V)
        
        weight = query_embed @ key_embed.transpose(-1, -2) # (N, H, S, S)
        if mask:
            weight = weight.masked_fill(mask, -torch.inf)
        weight = weight / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32)) # N, H, S, S
        weight = F.softmax(weight, dim=-1) # (N, H, S, S)
        attns = weight @ value_embed # (N, H, S, V)
        
        return self.scores(attns.transpose(1, 2).flatten(start_dim=-2)) # flatten: (N, S, H * V) and H * V = embed_dim => valid
