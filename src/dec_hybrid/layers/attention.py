import torch, torch.nn as nn, torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class LLAttentionBlock(nn.Module):
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
    def __init__(self, dim, num_heads=8, head_dim = 64, hkv_processor=None):
        super().__init__()
        assert dim % num_heads == 0, "The embedding dimension must be divisible by num_heads."

        # Core Parameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads if head_dim is None else head_dim
        self.scale = self.head_dim ** -0.5

        if hkv_processor is not None:
            # Learnable parameter for lookahead modulation
            self.alpha = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # Projection layers
        self.to_qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
        self.to_out = nn.Linear(self.head_dim * self.num_heads, dim)

        # Positional embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Flexible processors for HK and HV
        # If no processor is provided, defaults to standard decoder model
        self.hkv_processor = hkv_processor() if hkv_processor is not None else None



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

        # fixed weighting (helps?)
        # multihead_out = 0.5 * W @ V +   0.5 * (HV @ V.transpose(-2,-1) * W).sum(dim=-1, keepdim=True) * HV

        return multihead_out
    
    def _causal_attention(self, Q, K, V):
        """
        Q, K, V: (B, H, N, Hd)
   
        """

        # Scaled dot-product attention
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


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

        if self.hkv_processor is not None:
            HK, HV = self.hkv_processor(X)
            HK = HK.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            HV = HV.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            multihead_out = self._lookahead_attention(Q, K, V, HK, HV)
        else:
            multihead_out = self._causal_attention(Q, K, V)


        attn_output = multihead_out.transpose(1, 2).contiguous().view(B, N, -1)
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
