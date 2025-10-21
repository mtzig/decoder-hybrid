import torch, torch.nn as nn, torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from .resids import RMSNorm

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


        self.hkv_processor = False

        if hkv_processor:
            # Learnable parameter for lookahead modulation
            self.to_kvkv = nn.Linear(dim, self.head_dim * self.num_heads * 4, bias=False)
            self.hkv_processor = True


        # Projection layers
        self.to_qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
        self.to_out = nn.Linear(self.head_dim * self.num_heads, dim)

        # Positional embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Flexible processors for HK and HV
        # If no processor is provided, defaults to standard decoder model





    
    def _causal_attention(self, Q, K, V):
        """
        Q, K, V: (B, H, N, Hd)
   
        """

        # Scaled dot-product attention
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


    def chunked_parallel_implementation(self, Q, KK, VK, KV, VV, chunk_size=128):
        """
        A memory-efficient parallel implementation suitable for training.
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        device, dtype = Q.device, Q.dtype

        S0 = torch.eye(head_dim, device=device, dtype=dtype) * (head_dim ** 0.5)
        SK_state = S0.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_heads, 1, 1)
        SV_state = SK_state.clone()

        outputs_y = []
        outputs_sv = []

        # Process the sequence in chunks for memory efficiency
        for i in range(0, seq_len, chunk_size):
            chunk_end = i + chunk_size
            q_chunk = Q[:, :, i:chunk_end]
            kk_chunk = KK[:, :, i:chunk_end]
            vk_chunk = VK[:, :, i:chunk_end]
            kv_chunk = KV[:, :, i:chunk_end]
            vv_chunk = VV[:, :, i:chunk_end]

            # --- Parallel computation within the chunk ---
            # 1. Outer products for the chunk. Memory intensive but on a smaller tensor.
            ok_chunk = torch.einsum('bhni, bhnj -> bhnij', kk_chunk, vk_chunk)
            ov_chunk = torch.einsum('bhni, bhnj -> bhnij', kv_chunk, vv_chunk)

            # 2. Intra-chunk prefix sum (cumsum)
            SK_prefix_sum_chunk = torch.cumsum(ok_chunk, dim=2)
            SV_prefix_sum_chunk = torch.cumsum(ov_chunk, dim=2)

            # 3. Add state from the previous chunk to make the prefix sum global
            SK_chunk = SK_prefix_sum_chunk + SK_state.unsqueeze(2)
            SV_chunk = SV_prefix_sum_chunk + SV_state.unsqueeze(2)

            # --- Normalization and Output Calculation for the chunk---
            SK_norm = torch.norm(SK_chunk, p='fro', dim=(-2, -1), keepdim=True) + 1e-8
            SK_normalized_chunk = SK_chunk / SK_norm
            
            SV_norm = torch.norm(SV_chunk, p='fro', dim=(-2, -1), keepdim=True) + 1e-8
            SV_normalized_chunk = SV_chunk / SV_norm

            y_chunk = torch.einsum('bhni, bhnij -> bhnj', q_chunk, SK_normalized_chunk.transpose(-2,-1))
            
            outputs_y.append(y_chunk)
            outputs_sv.append(SV_normalized_chunk)

            # --- Update the state for the next chunk ---
            SK_state = SK_chunk[:, :, -1, :, :]
            SV_state = SV_chunk[:, :, -1, :, :]
        
        final_y = torch.cat(outputs_y, dim=2)
        final_sv = torch.cat(outputs_sv, dim=2)

        return final_y, final_sv

    def forward(self, X):
        B, N, D = X.size()

        # split last dim into evenly sized blocks
        QKV = self.to_qkv(X).chunk(3, dim=-1)

        # (batch, heads, sequence, head_dim)
        Q, K, V = map(lambda tensor: tensor.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), QKV)

        # Apply rotary embeddings
        # NOTE: might need to think more about this design choice


        if self.hkv_processor:

            KVKV = self.to_kvkv(X).chunk(4, dim=-1)

            # (batch, heads, sequence, head_dim)
            KK, VK, KV, VV = map(lambda tensor: tensor.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), KVKV)

            # OK = torch.einsum('bhni, bhnj -> bhnij', KK, VK)
            # OV = torch.einsum('bhni, bhnj -> bhnij', KV, VV)

            # print('create tensors')

            # S0 = torch.eye(self.head_dim, device=X.device, dtype=X.dtype) * (self.head_dim ** 0.5)

            # SK = torch.cumsum(OK, dim=2) + S0
            # SV = torch.cumsum(OV, dim=2) + S0

            # # SK = OK
            # # SV = OV

            # print('prefix sum')

            # SK_norm = torch.norm(SK, p='fro', dim=(-2, -1), keepdim=True) + 1e-8
            # SV_norm = torch.norm(SV, p='fro', dim=(-2, -1), keepdim=True) + 1e-8

            # print('compute norms')
            # SK = SK / SK_norm
            # SV = SV / SV_norm
            # Q = torch.einsum('bhni, bhnij -> bhnj', Q, SK.transpose(-2,-1))

            # print('compute Q')

            Q, SV = self.chunked_parallel_implementation(Q, KK, VK, KV, VV)
            print('lets seee')
            Q = self.rotary_emb.rotate_queries_or_keys(Q)
            K = self.rotary_emb.rotate_queries_or_keys(K)

            multihead_out = self._causal_attention(Q, K, V)
            # multihead_out = torch.einsum('bhni, bhnij -> bhnj', multihead_out, SV)

        else:
            Q = self.rotary_emb.rotate_queries_or_keys(Q)
            K = self.rotary_emb.rotate_queries_or_keys(K)
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
