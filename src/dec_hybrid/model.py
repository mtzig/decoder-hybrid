import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ALL the building blocks from your layers file
from layers import (
    LookaheadDecoderBlock, 
    LinearProjectionHKHV, 
    Mamba2HKHV,
    RMSNorm,
    SwiGLU
)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hkv_processor: nn.Module):
        super().__init__()
        self.attn = LookaheadDecoderBlock(
            dim=dim, 
            num_heads=num_heads, 
            hkv_processor=hkv_processor
        )
        self.ffn = SwiGLU(dim=dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class LookaheadTransformer(nn.Module):
    """
    A complete Transformer model composed of multiple TransformerBlocks.

    Args:
        vocab_size (int): The size of the vocabulary.
        dim (int): The embedding dimension.
        depth (int): The number of transformer blocks.
        num_heads (int): The number of attention heads.
        hkv_processor_factory (callable): A function or class that returns an 
                                          instance of the HK/HV processor.
    """
    def __init__(self, vocab_size: int, dim: int, depth: int, num_heads: int, hkv_processor_factory):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Create a stack of TransformerBlocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hkv_processor_factory()) 
            for _ in range(depth)
        ])
        
        self.final_norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        # (B, N) -> (B, N, D)
        x = self.token_emb(x)
        
        # Pass through all the transformer blocks
        for block in self.layers:
            x = block(x)
        
        # Final normalization and projection to logits
        x = self.final_norm(x)
        logits = self.to_logits(x)
        return logits
    

if __name__ == '__main__':
    # --- Model Parameters ---
    vocab_size = 10000
    dim = 512
    depth = 6
    num_heads = 8
    seq_len = 128
    batch_size = 4

    # --- Experiment 1: Using a simple Linear Projection for HK/HV ---
    print("🚀 Experiment 1: Linear Projection HK/HV Processor")
    # We use a lambda function as a factory to create a new instance for each layer
    linear_factory = lambda: LinearProjectionHKHV(dim=dim)
    model_linear = LookaheadTransformer(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        hkv_processor_factory=linear_factory
    )

    # --- Experiment 2: Using Mamba2 for HK/HV ---
    print("\n🐍 Experiment 2: Mamba2 HK/HV Processor")
    mamba_factory = lambda: Mamba2HKHV(dim=dim)
    model_mamba = LookaheadTransformer(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        hkv_processor_factory=mamba_factory
    )



    # --- Test forward pass ---
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test the linear projection model
    output_linear = model_linear(dummy_input)
    print(f"\nOutput shape from Linear model: {output_linear.shape}")
    assert output_linear.shape == (batch_size, seq_len, vocab_size)

    # Test the Mamba model
    # output_mamba = model_mamba(dummy_input)
    # print(f"Output shape from Mamba model: {output_mamba.shape}")
    # assert output_mamba.shape == (batch_size, seq_len, vocab_size)



    print("\n✅ All models initialized and tested successfully!")
