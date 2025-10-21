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

        if hkv_processor is not None:
            # Learnable parameter for lookahead modulation
            self.alpha = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_heads, 1, 1))


        # Projection layers
        self.to_qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
        self.to_out = nn.Linear(self.head_dim * self.num_heads, dim)

        # Positional embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Flexible processors for HK and HV
        # If no processor is provided, defaults to standard decoder model
        self.hkv_processor = hkv_processor() if hkv_processor is not None else None

        self.hknorm = RMSNorm(self.head_dim)
        self.hvnorm = RMSNorm(self.head_dim)

    def _lookahead_attention(self, Q, K, V, HK, HV):
        '''
        hk encodes info about keys
        hv encodes info about values
        '''
        
        _, _, N, _ = Q.size()

        # normalize HK/HV a bit for stability
        HK = self.hknorm(HK)
        HV = self.hvnorm(HV)

        # Modifications, Use SiLU nonlinearity, scale all scores by head_dim^-0.5

        causal_mask = torch.tril(torch.ones((N, N), device=Q.device))

        # attn_scores = Q @ K.transpose(-2,-1) * self.scale + self.beta * F.relu( HK @ K.transpose(-2,-1) * (Q * HK).sum(dim=-1, keepdim=True) * self.scale)

        attn_scores = (1 + self.beta * F.relu( HK @ K.transpose(-2,-1))) * (Q @ K.transpose(-2,-1)) * self.scale

        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # weights
        W = torch.softmax(attn_scores, dim=-1)


        # weird hybrid output...
        # TODO: try to write some Triton kernel for this lol
        multihead_out = W @ V + self.alpha * (HV @ V.transpose(-2,-1) * W).sum(dim=-1, keepdim=True) * HV
        
        
        # multihead_out =  ((W + self.alpha * (HV @ V.transpose(-2,-1)))) @ V
  
        # fixed weighting (helps?)
        # multihead_out = 0.5 * W @ V +   0.5 * (HV @ V.transpose(-2,-1) * W).sum(dim=-1, keepdim=True) * HV
        print(f'params alpha: {self.alpha}, beta: {self.beta}')
        return multihead_out
    # def __init__(self, dim, num_heads=8, head_dim=None, hkv_processor=None, lowrank_R=8):
    #     super().__init__()
    #     assert dim % num_heads == 0, "The embedding dimension must be divisible by num_heads."
    #     self.num_heads = num_heads
    #     self.head_dim  = (dim // num_heads) if head_dim is None else head_dim
    #     self.scale = self.head_dim ** -0.5

    #     # Projection layers
    #     self.to_qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
    #     self.to_out = nn.Linear(self.head_dim * self.num_heads, dim)

    #     # Positional embeddings
    #     self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    #     # Flexible processors for HK and HV
    #     # If no processor is provided, defaults to standard decoder model
    #     self.hkv_processor = hkv_processor() if hkv_processor is not None else None

    #     # === Low-rank logit update (query-dependent) ===
    #     self.lowrank_R = lowrank_R
    #     H, d, R = self.num_heads, self.head_dim, self.lowrank_R

    #     # Per-head bases U, V  (initialized small to keep extra logits tiny)
    #     self.lr_U = nn.Parameter(torch.randn(H, d, R) / (d ** 0.5))
    #     self.lr_V = nn.Parameter(torch.randn(H, d, R) / (d ** 0.5))

    #     # Context -> [u_scale, v_scale] per token/head
    #     self.hk_norm   = RMSNorm(self.head_dim)
    #     self.ctx_uv_mlp = nn.Linear(self.head_dim, 2*R)

    #     # ReZero gate (per head), starts at zero => identical to baseline
    #     self.lr_gate = nn.Parameter(torch.zeros(H))

    #     # Optional dropout on attention probs (match your block’s config if you have it)
    #     self.attn_dropout = nn.Dropout(p=0.0)

    @torch.no_grad()
    def _causal_mask(self, T, device, dtype):
        # Lower-triangular causal mask as additive -inf (keeps logits linear-friendly)
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def _lookahead_attention_yolo2(self, Q, K, V, HK, HV):
        """
        Low-rank, query-dependent metric update on logits.
        Inputs (per-head tensors):
          Q, K, V: [B, H, T, d]
          HK:      [B, H, T, d]  (global/context sequence aligned with time)
          HV:      [B, H, T, d]  (unused here; kept for API compatibility)
        Returns:
          multihead_out: [B, H, T, d]
        """
        B, H, T, d = Q.shape
        assert H == self.num_heads and d == self.head_dim, "unexpected Q/K/V shape"

        # ---- Baseline logits ----
        # (Assumes RoPE has already been applied consistently to Q and K outside.)
        # base_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,T,T]

        # # ---- Context-derived low-rank query-dependent term ----
        # # Normalize context and get per-token scales u_s, v_s  in R^R
        # # Shapes: u_s, v_s -> [B,H,T,R]
        # hk = self.hk_norm(HK)
        # u_s, v_s = torch.chunk(self.ctx_uv_mlp(hk), 2, dim=-1)  # [B,H,T,R] each
        # # keep scales bounded
        # u_s = torch.tanh(u_s)
        # v_s = torch.tanh(v_s)

        # # Project Q and K into low-rank bases per head:
        # # Q_U: [B,H,T,R], K_V: [B,H,T,R]
        # Q_U = torch.einsum('bhtd,hdr->bhtr', Q, self.lr_U)
        # K_V = torch.einsum('bhtd,hdr->bhtr', K, self.lr_V)  # cacheable across decode steps

        # # Extra logits: sum_r ( (q_u * u_s)_t * (k_v * v_s)_j )
        # # -> [B,H,T,T]
        # # (Broadcast u_s over j dimension; v_s multiplies K_V per j.)
        # extra_logits = torch.einsum('bhtr,bhtr,bhjr->bhtj', Q_U, u_s, K_V * v_s)

        # # ReZero gate α per head, broadcast to [B,H,T,T]
        # alpha = self.lr_gate.view(1, H, 1, 1)
        # logits = base_logits + alpha * (extra_logits * self.scale)

        # # ---- Causal masking ----
        # logits = logits + self._causal_mask(T, logits.device, logits.dtype)

        # # ---- Attention weights ----
        # attn = torch.softmax(logits, dim=-1)
        # attn = self.attn_dropout(attn)

        # # ---- Standard value aggregation ----
        # out = torch.matmul(attn, V)  # [B,H,T,d]

        # return out

    # def __init__(self, dim, num_heads=8, head_dim=64, hkv_processor=None,
    #              la_rank=16, la_bias_dim=None):
    #     super().__init__()
    #     assert dim % num_heads == 0, "The embedding dimension must be divisible by num_heads."

    #     # Core Parameters
    #     self.num_heads = num_heads
    #     self.head_dim = dim // num_heads if head_dim is None else head_dim
    #     self.scale = self.head_dim ** -0.5

    #     if hkv_processor is not None:
    #         # (kept from your code; no longer used by the new path, safe to keep/remove)
    #         self.alpha = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    #     # Projection layers
    #     self.to_qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
    #     self.to_out = nn.Linear(self.head_dim * self.num_heads, dim)

    #     # Positional embeddings
    #     self.rotary_emb = RotaryEmbedding(self.head_dim)

    #     # Flexible processors for HK and HV
    #     self.hkv_processor = hkv_processor() if hkv_processor is not None else None

    #     # ========= Lookahead (1) LoRA-style updates to K/V from HK/HV =========
    #     # Rank for low-rank adapters
    #     self.la_rank = la_rank
    #     # Dim for the logit-bias projection space
    #     self.la_bias_dim = la_bias_dim if la_bias_dim is not None else min(64, self.head_dim)

    #     # Normalizers for HK/HV before projecting (shared across heads)
    #     self.hk_ln = nn.LayerNorm(self.head_dim)
    #     self.hv_ln = nn.LayerNorm(self.head_dim)

    #     # LoRA adapters (shared across heads): HK -> ΔK, HV -> ΔV
    #     self.hk_B = nn.Linear(self.head_dim, self.la_rank, bias=False)   # HK -> r
    #     self.hk_A = nn.Linear(self.la_rank,  self.head_dim, bias=False)  # r  -> d_h
    #     self.hv_B = nn.Linear(self.head_dim, self.la_rank, bias=False)   # HV -> r
    #     self.hv_A = nn.Linear(self.la_rank,  self.head_dim, bias=False)  # r  -> d_h

    #     # Inits: B legs Xavier, A legs zero (so ΔK/ΔV start at 0)
    #     nn.init.xavier_uniform_(self.hk_B.weight)
    #     nn.init.xavier_uniform_(self.hv_B.weight)
    #     nn.init.zeros_(self.hk_A.weight)
    #     nn.init.zeros_(self.hv_A.weight)

    #     # Gates for K/V updates (start at 0 => exact baseline initially)
    #     self.gk   = nn.Parameter(torch.tensor(0.0))
    #     self.gv   = nn.Parameter(torch.tensor(0.0))

    #     # ========= Lookahead (2) Additive HK-derived logit bias =========
    #     self.W_qb = nn.Linear(self.head_dim, self.la_bias_dim, bias=False)
    #     self.W_kb = nn.Linear(self.head_dim, self.la_bias_dim, bias=False)
    #     nn.init.xavier_uniform_(self.W_qb.weight)
    #     nn.init.xavier_uniform_(self.W_kb.weight)

    #     # Gate for the bias path (start at 0 => baseline)
    #     self.beta = nn.Parameter(torch.tensor(0.0))







    # def _lookahead_attention(self, Q, K, V, HK, HV):
    #     """
    #     Implements:
    #     (1) LoRA-style updates to K and V from HK/HV (zero-init gated)
    #     (2) Additive logit bias from HK (zero-init gated)
    #     Shapes: all [B, H, N, d_h]
    #     """
    #     B, H, N, d_h = Q.size()

    #     # ----- (1) LoRA-style updates -----
    #     # normalize HK/HV a bit for stability
    #     hk_n = self.hk_ln(HK)
    #     hv_n = self.hv_ln(HV)

    #     # delta_k, delta_v: [B,H,N,d_h]
    #     delta_k = self.hk_A(self.hk_B(hk_n))
    #     delta_v = self.hv_A(self.hv_B(hv_n))

    #     # gated updates (broadcast scalar gates)
    #     Kp = K + self.gk * delta_k
    #     Vp = V + self.gv * delta_v

    #     # ----- base logits from updated keys -----
    #     attn_scores = torch.matmul(Q, Kp.transpose(-2, -1)) * self.scale

    #     # ----- (2) additive HK-derived logit bias (cheap) -----
    #     q_bias  = self.W_qb(Q)              # [B,H,N,d_b]
    #     hk_bias = self.W_kb(hk_n)           # [B,H,N,d_b]
    #     bias = torch.matmul(q_bias, hk_bias.transpose(-2, -1)) / (hk_bias.size(-1) ** 0.5)
    #     attn_scores = attn_scores + self.beta * bias

    #     # causal mask
    #     causal_mask = torch.tril(torch.ones((N, N), device=Q.device, dtype=torch.bool))
    #     attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

    #     # weights
    #     W = torch.softmax(attn_scores, dim=-1)

    #     # output with updated values
    #     multihead_out = torch.matmul(W, Vp)
    #     return multihead_out
    
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
