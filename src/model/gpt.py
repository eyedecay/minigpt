import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .block import TransformerBlock
from .layers import LayerNorm

class GPTModel(nn.Module):
    """
    GPT Transformer with token embeddings, stack of blocks, final normalization, and logits

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer 
        context_length (int):
        drop_emb (nn.Dropout): dropout
        trf_blocks (nn.ModuleList): transformer blocks
        final_norm (LayerNorm): normalization
        out_head (nn.Linear): projecting hidden states to logits
    """
    def __init__(self, cfg):
        """
        Initializes GPT Model


        Args:
            cfg (Dict): Config dictionary
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.context_length = cfg["context_length"]
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)
    
    def forward(self, in_idx, past_kv_list = None):
        """
        Forward pass

        Args:
            in_idx (torch.Tensor): input token indices shape (batch_size, seq_len)
            past_kv_list (list, optional): cached key value pairs from previous forward passes

        Returns:
            (torch.Tensor): logits 
            (list): new_kv_list (updated cache)
        """
        x = self.tok_emb(in_idx)
        x = self.drop_emb(x)

        new_kv_list = []
        #processes output
        for i, block in enumerate(self.trf_blocks):
            past_kv = past_kv_list[i] if past_kv_list is not None else None 
            x, new_kv = block(x, past_kv)
            new_kv_list.append(new_kv)
        
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits, new_kv_list

