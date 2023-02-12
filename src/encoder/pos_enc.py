import numpy as np
import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, 
                 max_seq_len: float, 
                 embed_dim: float,
                 device: str="cuda") -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.pos_table = self._get_pos_enc_table()
        self._device = device
    
    def _get_pos_angle(self, position, embed_dim):
        return np.array([position / (10000 ** (2 * (i // 2) / embed_dim)) for i in range(embed_dim)], dtype=np.float32) # 1 x E
    
    def _get_pos_enc_table(self):
        pos_tab = np.array([self._get_pos_angle(i, self.embed_dim) for i in range(self.max_seq_len)]) # S x E
        pos_tab[::2] = np.sin(pos_tab[::2])
        pos_tab[1::2] = np.cos(pos_tab[1::2])
        return torch.tensor(pos_tab, dtype=torch.float32, device=self._device).unsqueeze(0) # 1 x S x E
    
    def forward(self, X: torch.Tensor):
        return X + self.pos_table[:, X.shape[1], :]