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
        src_padding_index = 2
        tgt_padding_index = 3
        
        inputs = torch.randint(10, size=(batch_size, in_seq_len))
        outputs = torch.randint(10, size=(batch_size, out_seq_len))
        
        model = AttentionV1(embed_dim=embed_dim, 
                            num_head=num_head, 
                            src_padding_idx=src_padding_index,
                            tgt_padding_idx=tgt_padding_index,
                            src_vocab_len=10,
                            tgt_vocab_len=10)
        pred = model(inputs, outputs)
        
        self.assertEquals(pred.shape, torch.Size((batch_size, out_seq_len, 10)))
        
    def test_match_num_params(self):
        std_transformer = nn.Transformer(512, 8)
        custom_transformer = AttentionV1(src_vocab_len=10, 
                                         tgt_vocab_len=10, 
                                         embed_dim=512, 
                                         num_head=8, 
                                         src_padding_idx=2, 
                                         tgt_padding_idx=3)
        
        std_num_params = sum(p.numel() for p in std_transformer.parameters() if p.requires_grad)
        custom_num_params_encoder = sum(p.numel() for p in custom_transformer.encoder_stack.parameters() if p.requires_grad)
        custom_num_params_decoder = sum(p.numel() for p in custom_transformer.decoder_stack.parameters() if p.requires_grad)
        custom_num_params_encoder_norm = sum(p.numel() for p in custom_transformer.norm_encoder.parameters() if p.requires_grad)
        custom_num_params_decoder_norm = sum(p.numel() for p in custom_transformer.norm_decoder.parameters() if p.requires_grad)
        
        self.assertEqual(std_num_params, custom_num_params_encoder + custom_num_params_decoder + custom_num_params_encoder_norm + custom_num_params_decoder_norm)
