import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from .model import LLTransformer
from transformers.modeling_outputs import CausalLMOutputWithPast
from .layers.HKHV import LinearProjectionHKHV, Mamba2HKHV, AR1HKHV
from transformers import AutoConfig, AutoModelForCausalLM


class LLTransformerConfig(PretrainedConfig):
    model_type = "LLTransformer"

    def __init__(self, vocab_size=30522, dim=512, depth=6, num_heads=8, head_dim=64, hkv_processor_factory='decoder-only',**kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hkv_processor_factory = hkv_processor_factory

class  LLTransformerForCausalLM(PreTrainedModel):
    config_class =LLTransformerConfig

    def __init__(self, config: LLTransformerConfig):
        super().__init__(config)

        if config.hkv_processor_factory == 'decoder-only':
            hkv_processor_factory = None
        elif config.hkv_processor_factory == 'linear':
            hkv_processor_factory = lambda: LinearProjectionHKHV(dim=config.dim, num_heads=config.num_heads, head_dim=config.head_dim)
        elif config.hkv_processor_factory == 'mamba2':
            hkv_processor_factory = lambda: Mamba2HKHV(dim=config.dim, num_heads=config.num_heads, head_dim=config.head_dim)
        elif config.hkv_processor_factory == 'ar1':
            hkv_processor_factory = lambda: AR1HKHV(dim=config.dim, num_heads=config.num_heads, head_dim=config.head_dim)
        else:
            raise ValueError(f"Unknown hkv_processor_factory: {config.hkv_processor_factory}")
        
        self.transformer = LLTransformer(
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            hkv_processor_factory=hkv_processor_factory
        )

    def forward(
        self,
        input_ids,
        attention_mask = None,
        labels= None,
        **kwargs):
        """
        computes forward conforming to Huggingface's transformers library
        """

        logits = self.transformer(input_ids)  # or self.transformer(input_ids, attention_mask=attention_mask)

        # loss = None
        # if labels is not None:
        #     # Standard causal LM shift
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
        #                     shift_labels.view(-1), ignore_index=-100, reduce='sum')

        # return loss, logits
        return logits
        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=None,
        #     hidden_states=None,
        #     attentions=None,
        # )
    
AutoConfig.register("LLTransformer", LLTransformerConfig)
AutoModelForCausalLM.register(LLTransformerConfig, LLTransformerForCausalLM)