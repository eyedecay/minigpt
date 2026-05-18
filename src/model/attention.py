import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.model.RoPE import RotaryEmbedding, apply_RoPE
from src.model.layers import LinearLayer


class MultiHeadAttention(nn.Module):
    """
    The MultiHeadAttention mechanism in Transformer Architecture. 
    It projects query, key, and value vectors, splitting them across multiple heads
    Computes scaled dot-product attention
    Concatenates and projects result to output dimensions

    Attributes:
        d_out (int): dimension of attention layer output
        num_heads (int): number of attention heads
        head_dim (int): dimension of individual head
        W_query (nn.Linear): projects input into query vectors
        W_key (nn.Linear): projects input into key vectors
        W_value (nn.Linear): projects input into value vectors
        out_proj (nn.Linear): final projection of all heads
        dropout (nn.Dropout): dropout
        mask (torch.Tensor): prevents attention to future tokens (Causal self-attention)
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialization

        Args:
            d_in (int): input dimensionality
            d_out (int): output dimensionality
            context_length (int): max sequence length
            dropout (float): dropout probability
            num_heads (int): number of heads
            qkv_bias (bool, optional): include bias or not in projections. Defaults to False.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) 
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, context_length)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, past_kv = None):
        """
        forward pass of attention

        Args:
            x (torch.Tensor): input tensor shape (batch_size, num_tokens, d_in)

        Returns:
            torch.Tensor: output tensor with attention and projected features. 
        """
        batch_size, num_tokens, d_in = x.shape

        #(batch_size, tokens, d_out)
        keys = self.W_key(x)  
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #KV Cache
        past_len = past_kv[0].shape[2] if past_kv is not None else 0

        #RoPE Embeddings
        cos, sin = self.rope(queries, num_tokens, start_pos = past_len)
        queries = apply_RoPE(queries, cos, sin)
        keys = apply_RoPE(keys, cos, sin)

        if past_kv is not None:
            past_keys, past_values = past_kv 
            keys = torch.cat([past_keys, keys], dim = 2)
            values = torch.cat([past_values, values], dim = 2)

        #Compare a query with all keys to see similarity. (the dot product)
        attn_scores = queries @ keys.transpose(2, 3)  

        total_len = keys.shape[2]
        mask_bool = self.mask.bool()[past_len:past_len + num_tokens, :total_len]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  

        return context_vec, (keys, values)

 