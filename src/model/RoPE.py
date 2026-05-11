import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (modifies queries and keys in attention using rotations which encodes position) happens before attention

    """
    def __init__(self, dim, max_seq_len=1024):
        """
        Initializes RoPE.

        Args:
            dim (int): embedding dimension of queries and keys
            max_seq_len (int, optional): maximum sequence length. Defaults to 1024.
        """
        super().__init__()
        half_dim = dim // 2
        #Computes the rotation speeds for different dimensions (lower dimension = rotate slowly and vice versa)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float()/dim))
        positions = torch.arange(max_seq_len, dtype = torch.float)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_encoded", angles.cos())
        self.register_buffer("sin_encoded", angles.sin())
    
    def forward(self, x, seq_len, start_pos = 0):
        """
        Gets RoPE values for given splice in sequence

        Args:
            x (torch.Tensor): input Tensor
            seq_len (int): length of sequence
            start_pos (int, optional): starting position (changes and used for autoregressive generation). Defaults to 0.

        Returns:
            _type_: _description_
        """
        cos = self.cos_encoded[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_encoded[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        cos = cos.repeat_interleave(2, dim = -1)
        sin = sin.repeat_interleave(2, dim = -1)
        return cos, sin

def rotate_half_negate(x):
    """
    Helper function for RoPE. splits last dimension into two halves and swaps them. 90 degree rotation in space

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: rotated tensor (same shape) (..., dim)
    """
    mid = x.shape[-1] // 2
    first_half, second_half = x[..., :mid], x[...,mid:]
    return torch.cat((-second_half, first_half), dim = -1)

def apply_RoPE(query_or_key, cos_values, sin_values):
    """

    Args:
        query_or_key (torch.Tensor): query or key tensor
        cos_values (torch.Tensor): cosine encodings
        sin_values (torch.Tensor): sin encodings

    Returns:
        _type_: _description_
    """
    return query_or_key * cos_values + rotate_half_negate(query_or_key) * sin_values


        