import torch
import torch.nn as nn

def _as_btH(x):
    if x.ndim == 2:  # (T,H) -> (1,T,H)
        return x.unsqueeze(0)
    assert x.ndim == 3, f"expected (B,T,H) or (T,H), got {x.shape}"
    return x

@torch.no_grad()
def _left_pad_like(x, pad_T, fill):
    # x: (B,T,H) -> (B,T+pad_T,H) where the left pad is `fill`
    B, T, H = x.shape
    pad = torch.full((B, pad_T, H), fill, dtype=x.dtype, device=x.device)
    return torch.cat([pad, x], dim=1)

def ar1_scan_parallel_safe(a_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    """
    Parallel AR(1) prefix without in-place slice assignment:
        h_t = a_t * h_{t-1} + b_t,  h_0 = 0, elementwise over H.
    Shapes: a_t, b_t in (B,T,H) or (T,H). Returns h in (B,T,H).
    Complexity: O(T log T). Compatible with torch.compile / AOTAutograd.
    """
    a = _as_btH(a_t).contiguous()  # (B,T,H)
    c = _as_btH(b_t).contiguous()  # (B,T,H)
    B, T, H = c.shape

    A = a
    C = c
    step = 1
    # binary-lifting scan: combine (A,C) with left-shifted prefixes each round
    while step < T:
        # Left-shifted (by `step`) versions with identity/zero padding on the left
        A_shift = _left_pad_like(A[:, :-step, :], pad_T=step, fill=1.0)  # identity for A
        C_shift = _left_pad_like(C[:, :-step, :], pad_T=step, fill=0.0)  # zero for C

        # Combine out-of-place (no slicing writes)
        # (A,C) ⊗ (A_shift, C_shift) = (A*A_shift, A*C_shift + C)
        C = A * C_shift + C
        A = A * A_shift
        step <<= 1
    return C  # inclusive prefixes: h_t

class AR1ScanTV(nn.Module):
    """Time-varying AR(1) with safe parallel scan."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, stable='tanh', leak=0.0):
        super().__init__()
        self.to_ab = nn.Linear(d_in, 1 + d_hidden, bias=True)  # -> [a_t | b_t]
        self.Wy = nn.Linear(d_hidden, d_out, bias=True)
        assert stable in ('tanh', 'sigmoid')
        self.stable = stable
        self.leak = leak  # e.g., 0.01 to enforce |a_t|<=0.99

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_btH(x)
        ab = self.to_ab(x)                    # (B,T,1+H)
        a_raw = ab[..., :1]                   # (B,T,1)
        b_t   = ab[..., 1:]                   # (B,T,H)

        if self.stable == 'tanh':
            a_t = torch.tanh(a_raw)
        else:
            a_t = torch.sigmoid(a_raw) * 2 - 1  # (-1,1)
        if self.leak > 0:
            a_t = a_t.clamp(min=-1 + self.leak, max=1 - self.leak)
        a_t = a_t.expand_as(b_t)              # (B,T,H)

        h = ar1_scan_parallel_safe(a_t, b_t)  # (B,T,H)
        return self.Wy(h)
