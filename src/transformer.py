import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from decoder.decoder import Decoder
from encoder.encoder import Encoder


class AttentionV1(pl.LightningModule):
    def __init__(self, 
                 src_vocab_len: int,
                 tgt_vocab_len: int,
                 embed_dim: int, 
                 num_head: int,
                 dim_ffn: int=2048,
                 num_stack: int=6) -> None:
        super().__init__()
        self.src_vocab_len = src_vocab_len
        self.tgt_vocab_len = tgt_vocab_len
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dim_ffn = dim_ffn
        self.num_stack = num_stack
        
        self.src_embedding = nn.Embedding(self.src_vocab_len, self.embed_dim)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_len, self.embed_dim)
        self.encoder_stack = nn.ModuleList(Encoder(embed_dim, num_head, dim_ffn) for _ in range(self.num_stack))
        self.decoder_stack = nn.ModuleList(Decoder(embed_dim, num_head, dim_ffn) for _ in range(self.num_stack))
        self.norm_decoder = nn.LayerNorm(self.embed_dim)
        self.norm_encoder = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, self.tgt_vocab_len)
        
    def forward(self, inputs, outputs):
        inputs = self.src_embedding(inputs)
        outputs = self.src_embedding(outputs)
        # encoder forwards
        for encoder in self.encoder_stack:
            inputs = encoder(inputs)
        inputs = self.norm_encoder(inputs)
            
        # decoder forwards
        for decoder in self.decoder_stack:
            outputs = decoder(inputs, outputs)
        
        return self.linear(self.norm_decoder(outputs))
    
    def training_step(self, train_batch, batch_idx):
        src, tgt = train_batch
        label = F.one_hot(tgt, num_classes=self.tgt_vocab_len).float()
        pred = self.forward(src, tgt)
        loss = F.cross_entropy(pred, label, label_smoothing=0.1)
        self.log("train_loss", loss)
        return loss
    
    def _get_lr_scale(self, step: int):
        step_num = step + 1
        return self.embed_dim ** (-0.5) * min(step_num ** (-0.5), step_num * (4000 ** (-1.5)))
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self._get_lr_scale(0), betas=(0.9, 0.98), eps=1e-9)
        scheduler = LambdaLR(optimizer, self._get_lr_scale)
        return dict(optimizer=optimizer,
                    lr_scheduler=scheduler)
