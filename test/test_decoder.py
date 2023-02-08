import torch
import torch.nn as nn
import unittest

from decoder.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def test_decoder(self):
        embed_dim = 16
        num_head = 4
        seq_len = 33
        batch_size = 6
        
        inputs = torch.rand(batch_size, seq_len, embed_dim)
        
        decoder = Decoder(embed_dim, num_head)
        outputs = decoder(inputs, inputs)
        
        self.assertEqual(outputs.shape, inputs.shape)
        
    def test_match_num_params(self):
        std_decoder = nn.TransformerDecoderLayer(512, 8)
        custom_decoder = Decoder(512, 8)
        
        std_num_params = sum(p.numel() for p in std_decoder.parameters() if p.requires_grad)
        custom_num_params = sum(p.numel() for p in custom_decoder.parameters() if p.requires_grad)
        
        self.assertEqual(std_num_params, custom_num_params)
