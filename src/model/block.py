import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .attention import MultiHeadAttention
from .layers import LayerNorm, FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block with layer norm, multi-head self-attention, residual connection, layernorm, ffn, and residual connection

    Attributes:
        att (MultiHeadAttention): self-attention Module
        ff (FeedForward): position-wise ffn
        norm1 (LayerMorm): layer normalization before attention
        norm2 (LayerNorm): layer normalization before ffn
        drop_shortut (nn.Dropout): dropout to residuals
    """
    def __init__(self, cfg):
        """
        Initializes transformer block

        Args:
            cfg (dict): Config dictionary
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias= cfg["qkv_bias"],
        )
        #In one block: LayerNorm, MultiHead Attention, Dropout, LayerNorm2, FeedForward, Dropout
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x, past_kv = None):
        """


        Args:
            x (torch.Tensor): input tensor 
            past_kv (optional): cached key-value pairs. Defaults to None.

        Returns:
            _type_: _description_
        """
        shortcut = x

        x = self.norm1(x)
        attention_out, new_kv = self.att(x, past_kv)
        x = self.drop_shortcut(attention_out)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, new_kv
