import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from decoder import Decoder
from encoder import Encoder, PositionalEncoder


class AttentionV1(pl.LightningModule):
    def __init__(self, 
                 src_vocab_len: int,
                 tgt_vocab_len: int,
                 embed_dim: int, 
                 num_head: int,
                 src_padding_idx: int,
                 tgt_padding_idx: int,
                 max_src_seq_len: int=100, 
                 max_tgt_seq_len: int=100,
                 dim_ffn: int=2048,
                 num_stack: int=6) -> None:
        super().__init__()
        self.src_vocab_len = src_vocab_len
        self.tgt_vocab_len = tgt_vocab_len
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dim_ffn = dim_ffn
        self.num_stack = num_stack
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        
        self.src_embedding = nn.Embedding(self.src_vocab_len, self.embed_dim)
        self.src_pos_embedding = PositionalEncoder(max_seq_len=self.max_src_seq_len, 
                                                   embed_dim=self.embed_dim)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_len, self.embed_dim)
        self.tgt_pos_embedding = PositionalEncoder(max_seq_len=self.max_tgt_seq_len,
                                                   embed_dim=self.embed_dim)
        self.encoder_stack = nn.ModuleList(Encoder(embed_dim, num_head, dim_ffn) for _ in range(self.num_stack))
        self.decoder_stack = nn.ModuleList(Decoder(embed_dim, num_head, dim_ffn) for _ in range(self.num_stack))
        self.norm_decoder = nn.LayerNorm(self.embed_dim)
        self.norm_encoder = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(self.embed_dim, self.tgt_vocab_len)
        self.train_loss = None
        self.train_bleu = None
        
    def _get_src_mask_pad(self, seq: torch.Tensor):
        N, S = seq.shape
        mask = torch.ones(N, S ,S, dtype=torch.bool)
        pad_pos = torch.argwhere(seq == self.src_padding_idx)
        mask[pad_pos[:, 0], pad_pos[:, 1]] = False
        mask[pad_pos[:, 0], :, pad_pos[:, 1]] = False
        return mask
    
    def _get_tgt_mask_pad(self, seq: torch.Tensor):
        N, S = seq.shape
        mask = torch.ones(N, S ,S, dtype=torch.bool)
        pad_pos = torch.argwhere(seq == self.tgt_padding_idx)
        mask[pad_pos[:, 0], pad_pos[:, 1]] = False
        mask[pad_pos[:, 0], :, pad_pos[:, 1]] = False
        return mask
    
    def _get_subsequent_mask(self, input_seq: torch.Tensor, output_seq: torch.Tensor):
        _, S_in = input_seq.shape
        _, S_out = output_seq.shape
        return (1 - torch.triu(torch.ones(1, S_out, S_in), diagonal=1)).bool()
        
    def forward(self, inputs, outputs):
        src_mask = self._get_src_mask_pad(inputs)
        tgt_mask = self._get_tgt_mask_pad(outputs)
        cross_attn_mask = self._get_subsequent_mask(inputs, outputs)
        inputs = self.src_embedding(inputs)
        inputs = self.src_pos_embedding(inputs)
        outputs = self.tgt_embedding(outputs)
        outputs = self.tgt_pos_embedding(outputs)
        print(inputs.shape, outputs.shape)
        # encoder forwards
        for encoder in self.encoder_stack:
            inputs = encoder(inputs, mask=src_mask)
        inputs = self.norm_encoder(inputs)
            
        # decoder forwards
        for decoder in self.decoder_stack:
            outputs = decoder(inputs, 
                              outputs, 
                              slf_attn_mask=tgt_mask, 
                              cross_attn_mask=cross_attn_mask)
            
        print(inputs.shape, outputs.shape)
        
        return self.linear(self.norm_decoder(outputs))
    
    def training_step(self, train_batch, batch_idx):
        src, tgt = train_batch
        label = F.one_hot(tgt, num_classes=self.tgt_vocab_len).float()
        pred = self.forward(src, tgt)
        loss = F.cross_entropy(pred, label, label_smoothing=0.1)
        # bleu_score = calc_bleu_score(src, tgt, self)
        self.train_loss = loss
        # self.train_bleu = bleu_score
        self.log("train_loss", loss)
        self.log("lr", self._get_lr_scale(self.global_step))
        # self.log("train_bleu_score", bleu_score)
        return loss
    
    def validation_step(self, val_batch, val_idx):
        src, tgt = val_batch
        label = F.one_hot(tgt, num_classes=self.tgt_vocab_len).float()
        pred = self.forward(src, tgt)
        loss = F.cross_entropy(pred, label, label_smoothing=0.1)
        # bleu_score = calc_bleu_score(src, tgt, self)
        self.log("val_loss", loss)
        # self.log("val_bleu_score", bleu_score)
        
    def on_train_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch}, train_loss {self.train_loss}")
    
    def _get_lr_scale(self, step: int):
        step_num = step + 1
        return self.embed_dim ** (-0.5) * min(step_num ** (-0.5), step_num * (4000 ** (-1.5)))
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self._get_lr_scale(0), betas=(0.9, 0.98), eps=1e-9)
        scheduler = LambdaLR(optimizer, self._get_lr_scale)
        return dict(optimizer=optimizer,
                    lr_scheduler=scheduler)
