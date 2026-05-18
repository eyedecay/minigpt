import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class LayerNorm(nn.Module):
    """
    Normalizes activations across feature dimension. Computes mean and variance then applies transformation with learnable scale and shift
    Ensures that all token's 768 values have mean = 0 variance = 1

    Attributes:
        eps (float): small constant added to variance (for stability purposes)
        scale (nn.Paramter): multi factor
        shift (nn.Paramter): bias
    """
    def __init__(self, emb_dim):
        """
        Initializes Layer Norm

        Args:
            emb_dim (int): size of feature dimension to normalize over
        """
        super().__init__()
        self.eps = 1e-5

        #Scale and Shift Weights in Layer Normalization are used to adjust the normalized output
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        """
        Applies normalization to input tensor

        Args:
            x (torch.Tensor): input tensor shape(..., emb_dim)

        Returns:
            torch.Tensor : normalized Tensor
        """
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """
    GELU from scratch activation function

    Attributes: 
        None
    """
    def __init__(self):
        """Initializes GELU"""
        super().__init__()

    def forward(self, x):
        """Computes GELU

        Args:
            x (torch.Tensor): input Tensor

        Returns:
            torch.Tensor: tensor with GELU Applied
        """
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715 * torch.pow(x,3))))

class LinearLayer(nn.Module):
    """
    Linear Layer from scratch. matrix multiplication of every row of x with every row of weight. nn.Parameter means it is learnable.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, d_in, d_out, bias = True):
        super().__init__()
        #Each weight is how strongly does the input activate a certain output (is there a relationship)
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    

class FeedForward(nn.Module):
    """
    Tranforms embedding dimension to Linear to higher space (4x) to GElU Linear back to 768 dimension space. After attention gets information, the FeedForward processes it. 

    Attributes:
        cfg (Dict): GPT Config
    """
    def __init__(self, cfg):
        """
        Initializes FeedForwad

        Args:
            cfg (Dict): GPT Config
        """
        super().__init__()
        #Linear Layer, GELU, Linear Layer
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)
    
    def forward(self, x):
        """
        

        Args:
            x (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        #calls self.layers 
        return self.layers(x)