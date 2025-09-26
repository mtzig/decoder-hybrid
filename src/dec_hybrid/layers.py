import torch, torch.nn as nn, torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from mamba_ssm import Mamba2

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

class LookaheadDecoderBlock(nn.Module):
    """
    A modified multi-head attention block that allows for flexible processing 
    of 'lookahead' key (HK) and value (HV) tensors.

    Args:
        dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        hk_processor (nn.Module, optional): A module to process the input tensor X 
                                            to generate HK. Defaults to nn.Identity.
        hv_processor (nn.Module, optional): A module to process the input tensor X 
                                            to generate HV. Defaults to nn.Identity.
    """
    def __init__(self, dim, num_heads=8, hkv_processor=None):
        super().__init__()
        assert dim % num_heads == 0, "The embedding dimension must be divisible by num_heads."

        # Core Parameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable parameter for lookahead modulation
        self.alpha = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # Projection layers
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Positional embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Flexible processors for HK and HV
        # If no processor is provided, it will just pass the input through (X -> X)
        self.hkv_processor = hkv_processor if hkv_processor is not None else nn.Identity()



    def _lookahead_attention(self, Q, K, V, HK, HV):
        '''
        hk encodes info about keys
        hv encodes info about values
        '''
        
        _, _, N, _ = Q.size()

        # Modifications, Use SiLU nonlinearity, scale all scores by head_dim^-0.5

        causal_mask = torch.tril(torch.ones((N, N), device=Q.device))

        attn_scores = Q @ K.transpose(-2,-1) * self.scale - F.silu( HK @ K.transpose(-2,-1) * (Q * HK).sum(dim=-1, keepdim=True) * self.scale)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # weights
        W = torch.softmax(attn_scores, dim=-1)

        # weird hybrid output...
        # TODO: try to write some Triton kernel for this lol
        multihead_out = W @ V - torch.sigmoid(self.alpha) * (HV @ V.transpose(-2,-1) * W).sum(dim=-1, keepdim=True) * HV

        return multihead_out

    def forward(self, X):
        B, N, D = X.size()

        # split last dim into evenly sized blocks
        QKV = self.to_qkv(X).chunk(3, dim=-1)

        # (batch, heads, sequence, head_dim)
        Q, K, V = map(lambda tensor: tensor.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), QKV)

        # Apply rotary embeddings
        # NOTE: might need to think more about this design choice
        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.rotary_emb.rotate_queries_or_keys(K)

        # COMPUTE ATTENTION SCORE HERE

        HK, HV = self.hkv_processor(X)
        HK = HK.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        HV = HV.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        multihead_out = self._lookahead_attention(Q, K, V, HK, HV)


        attn_output = multihead_out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.to_out(attn_output)
        return out

class CausalAttentionDecoderBlock(nn.Module):
    """
    A standard causal multi-head self-attention block using the same API
    structure as LookaheadDecoderBlock.

    Args:
        dim (int): The embedding dimension.
        num_heads (int): Number of attention heads.
        hkv_processor (nn.Module, optional): Kept for API parity; unused.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "The embedding dimension must be divisible by num_heads."

        # Core parameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5



        # Projections
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Positional embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def _causal_attention(self, Q, K, V, attn_bias=None):
        """
        Q, K, V: (B, H, N, Hd)
        attn_bias: optional tensor broadcastable to (B, H, N, N)
        """
        B, H, N, Hd = Q.shape

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Causal mask (allow attending to self and past only)
        causal_mask = torch.triu(
            torch.ones(N, N, device=attn_scores.device, dtype=torch.bool), diagonal=1
        )  # True above diagonal
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Optional additive bias (e.g., ALiBi, padding mask)
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N, Hd)
        return out

    def forward(self, X, attn_bias=None):
        """
        X: (B, N, D)
        attn_bias: optional tensor broadcastable to (B, H, N, N)
        """
        B, N, D = X.shape

        # Project and split into Q, K, V
        Q, K, V = self.to_qkv(X).chunk(3, dim=-1)  # each (B, N, D)

        # (B, H, N, Hd)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to Q and K
        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.rotary_emb.rotate_queries_or_keys(K)

        # Standard causal attention
        multihead_out = self._causal_attention(Q, K, V, attn_bias=attn_bias)

        # Merge heads
        attn_output = multihead_out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(attn_output)
        return out

class LinearProjectionHKHV(nn.Module):
    """
    A simple linear projection module to generate HK and HV from input tensor X.

    Args:
        dim (int): The embedding dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.to_h = nn.Linear(dim, dim*2, bias=False)

    def forward(self, X):
        HK,HV = self.to_h(X).chunk(2, dim=-1)
        return HK, HV
    

class Mamba2HKHV(nn.Module):
    """
    A mamba2 module to generate HK and HV from input tensor X.

    Args:
        dim (int): The embedding dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.mamba = Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2)
        self.to_h = nn.Linear(dim, dim*2, bias=False)

    def forward(self, X):
        H = self.mamba(X)
        HK,HV = self.to_h(H).chunk(2, dim=-1)
        return HK, HV
