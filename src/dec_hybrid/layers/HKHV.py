import torch, torch.nn as nn, torch.nn.functional as F
from mamba_ssm import Mamba2
from .ar1 import AR1ScanTV

class LinearProjectionHKHV(nn.Module):
    """
    A simple linear projection module to generate HK and HV from input tensor X.

    Args:
        dim (int): The embedding dimension.
    """
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.to_h = nn.Linear(dim, num_heads*head_dim*2, bias=False)

    def forward(self, X):
        HK,HV = self.to_h(X).chunk(2, dim=-1)
        return HK, HV
    

class Mamba2HKHV(nn.Module):
    """
    A mamba2 module to generate HK and HV from input tensor X.

    Args:
        dim (int): The embedding dimension.
    """
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.mamba = Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2)
        self.to_h = nn.Linear(dim, num_heads*head_dim*2, bias=False)

    def forward(self, X):
        H = self.mamba(X)
        HK,HV = self.to_h(H).chunk(2, dim=-1)
        return HK, HV
    


class AR1HKHV(nn.Module):
    """
    Time-varying AR(1) block with parallel scan.
    y_t = W_y h_t, where h_t = a_t ⊙ h_{t-1} + b_t, h_0=0
    a_t, b_t are produced from x_t.
    """
        
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.ar1 = AR1ScanTV(d_in=dim, d_hidden=num_heads*head_dim, d_out=num_heads*head_dim*2)

    def forward(self, X):
        HK,HV = self.ar1(X).chunk(2, dim=-1)
        return HK, HV






