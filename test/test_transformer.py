import torch
import torch.nn as nn
import unittest


from transformer import AttentionV1


class TestAttentionV1(unittest.TestCase):
    def test_shape(self):
        embed_dim = 16
        num_head = 4
        in_seq_len = 33
        out_seq_len = 20
        batch_size = 6
        
        inputs = torch.rand(batch_size, in_seq_len, embed_dim)
        outputs = torch.rand(batch_size, out_seq_len, embed_dim)
        
        model = AttentionV1(embed_dim, num_head)
        pred = model(inputs, outputs)
        
        self.assertEquals(pred.shape, outputs.shape)
        
    def test_match_num_params(self):
        std_transformer = nn.Transformer(512, 8)
        custom_transformer = AttentionV1(512, 8)
        
        std_num_params = sum(p.numel() for p in std_transformer.parameters() if p.requires_grad)
        custom_num_params = sum(p.numel() for p in custom_transformer.parameters() if p.requires_grad)
        
        self.assertEqual(std_num_params, custom_num_params)
