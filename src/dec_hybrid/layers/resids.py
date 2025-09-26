import torch, torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, ffn_hidden_mult: int = 4):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else int(ffn_hidden_mult * dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        fused_swiglu = F.silu(gate) * up
        return self.w2(fused_swiglu)