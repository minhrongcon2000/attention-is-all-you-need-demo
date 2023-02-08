import unittest

import torch
import torch.nn as nn

from attention.mh_attention import MultiHeadAttention


class TestMHAttention(unittest.TestCase):
    def test_shape(self):
        embed_dim = 16
        seq_len = 33
        batch_size = 6
        num_head = 4
        attn = MultiHeadAttention(embed_dim, num_head)
        
        inputs = torch.rand(batch_size, seq_len, embed_dim)
        
        attn_score = attn(inputs, inputs, inputs)
        self.assertEqual(attn_score.shape, (batch_size, seq_len, embed_dim))
        
    def test_shape_with_mask(self):
        embed_dim = 16
        seq_len = 33
        batch_size = 6
        num_head = 4
        attn = MultiHeadAttention(embed_dim, num_head)
        
        inputs = torch.rand(batch_size, seq_len, embed_dim)
        
        attn_score = attn(inputs, inputs, inputs, mask=True)
        self.assertEqual(attn_score.shape, (batch_size, seq_len, embed_dim))
        
    def test_violate_class_construction(self):
        embed_dim = 16
        num_head = 5
        self.assertRaises(AssertionError, MultiHeadAttention, embed_dim=embed_dim, num_head=num_head)
        
    def test_violate_forward_fn_with_dim2(self):
        embed_dim = 16
        num_head = 4
        key_dim = 5 
        value_dim = 5
        batch_size = 10
        seq_len = 20
        
        key, query, value = torch.rand(batch_size, seq_len), torch.rand(batch_size, seq_len), torch.rand(batch_size, seq_len)
        self.assertRaises(AssertionError, MultiHeadAttention(embed_dim, num_head), key, query, value)
        
    def test_violate_forward_fn_with_different_embedding(self):
        embed_dim = 16
        num_head = 4
        key_dim = 5 
        value_dim = 5
        batch_size = 10
        seq_len = 20
        
        key, query, value = torch.rand(batch_size, seq_len, key_dim), torch.rand(batch_size, seq_len, key_dim), torch.rand(batch_size, seq_len, value_dim)
        self.assertRaises(AssertionError, MultiHeadAttention(embed_dim, num_head), key, query, value, True)
        
    def test_match_num_params_with_standard_impl(self):
        # use `Attention is all you need` paper config
        std_head = nn.MultiheadAttention(512, 8)
        custom_head = MultiHeadAttention(512, 8)
        
        std_num_params = sum(p.numel() for p in std_head.parameters() if p.requires_grad)
        custom_num_params = sum(p.numel() for p in custom_head.parameters() if p.requires_grad)
        
        self.assertEqual(std_num_params, custom_num_params)
        
