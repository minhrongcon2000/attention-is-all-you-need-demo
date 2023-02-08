import torch
import torch.nn as nn
import unittest


from encoder.encoder import Encoder


class TestEncoder(unittest.TestCase):
    def test_encoder(self):
        embed_dim = 16
        num_head = 4
        seq_len = 33
        batch_size = 6
        
        inputs = torch.rand(batch_size, seq_len, embed_dim)
        
        encoder = Encoder(embed_dim, num_head)
        outputs = encoder(inputs)
        
        self.assertEqual(outputs.shape, inputs.shape)
        
    def test_match_num_params(self):
        std_encoder = nn.TransformerEncoderLayer(512, 8)
        custom_encoder = Encoder(512, 8)
        
        std_num_params = sum(p.numel() for p in std_encoder.parameters() if p.requires_grad)
        custom_num_params = sum(p.numel() for p in custom_encoder.parameters() if p.requires_grad)
        
        self.assertEqual(std_num_params, custom_num_params)
