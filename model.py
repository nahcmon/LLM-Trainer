import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.checkpoint import checkpoint

# Try to import FlashAttention (v2 or v3), fall back to PyTorch's optimized attention
FLASH_ATTENTION_AVAILABLE = False
flash_attn_func = None

try:
    # Try FlashAttention v3 first (if available)
    from flash_attn_3 import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print(">>> FlashAttention v3 enabled (fastest)")
except ImportError:
    try:
        # Fall back to FlashAttention v2
        from flash_attn import flash_attn_func
        FLASH_ATTENTION_AVAILABLE = True
        print(">>> FlashAttention v2 enabled (fastest)")
    except ImportError:
        # Check if PyTorch's optimized SDPA is available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print(">>> PyTorch optimized attention enabled (fast, built-in)")
        else:
            print(">>> Using manual attention implementation (slower)")

class FlashSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.n_head
        self.hd = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3*cfg.n_embd)
        self.o = nn.Linear(cfg.n_embd, cfg.n_embd)

        # Check if PyTorch's optimized SDPA is available (PyTorch 2.0+)
        self.use_pytorch_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        # Match GPT2Attention signature for compatibility with transformers>=5.0
        # Ignore encoder arguments (decoder-only), past_key_values for simplicity
        # FlashAttention handles causal masking internally

        x = hidden_states
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.hd)
        k = k.view(B, T, self.nh, self.hd)
        v = v.view(B, T, self.nh, self.hd)

        if FLASH_ATTENTION_AVAILABLE:
            # Use FlashAttention if available (fastest)
            out = flash_attn_func(q, k, v, causal=True)
        elif self.use_pytorch_sdpa:
            # Use PyTorch's optimized scaled_dot_product_attention (fast, available in PyTorch 2.0+)
            # This automatically uses Flash Attention or Memory-Efficient Attention if available
            q = q.transpose(1, 2)  # (B, nh, T, hd)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=0.0
            )
            out = out.transpose(1, 2)  # (B, T, nh, hd)
        else:
            # Fallback to manual scaled dot-product attention (slowest)
            q = q.transpose(1, 2)  # (B, nh, T, hd)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.hd ** 0.5))
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            out = attn @ v  # (B, nh, T, hd)
            out = out.transpose(1, 2)  # (B, T, nh, hd)

        attn_output = self.o(out.contiguous().view(B, T, C))

        # Return format matching GPT2Attention: (attn_output, attn_weights)
        # Always return tuple for consistency with transformers library
        attn_weights = None  # FlashAttention doesn't return attention weights
        return attn_output, attn_weights

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        # Gradient checkpointing wrapper for GPT2Block
        # Compatible with transformers>=5.0 API
        # For decoder-only models, ignore encoder-related arguments
        def custom_forward(hidden_states):
            return self.block(
                hidden_states,
                past_key_values=None,  # Don't use cache with gradient checkpointing
                cache_position=cache_position,
                attention_mask=attention_mask,
                encoder_hidden_states=None,  # Ignore encoder args for decoder-only
                encoder_attention_mask=None,  # Ignore encoder args for decoder-only
                output_attentions=output_attentions,
                **kwargs,
            )

        return checkpoint(custom_forward, hidden_states, use_reentrant=True)

def build_scratch_model(cfg, precision="fp16"):
    # Keep model in FP32 - autocast will handle precision during training
    # This is the recommended PyTorch approach for mixed precision

    # Build model config
    model_cfg = GPT2Config(
        vocab_size=cfg.get("vocab_size", 32000),
        n_positions=cfg.get("seq_length", 1024),
        n_ctx=cfg.get("seq_length", 1024),
        n_embd=cfg.get("hidden_size", 768),
        n_layer=cfg.get("n_layers", 12),
        n_head=cfg.get("n_heads", 12),
        n_inner=cfg.get("n_inner"),
        resid_pdrop=cfg.get("resid_pdrop", 0.1),
        embd_pdrop=cfg.get("embd_pdrop", 0.1),
        attn_pdrop=cfg.get("attn_pdrop", 0.1),
        activation_function=cfg.get("activation_function", "gelu_new"),
        layer_norm_epsilon=cfg.get("layer_norm_epsilon", 1e-5),
        initializer_range=cfg.get("initializer_range", 0.02),
        bos_token_id=2,
        eos_token_id=3
    )

    model = GPT2LMHeadModel(model_cfg)

    # Optionally replace with FlashAttention if available
    use_flash = cfg.get("use_flash_attention", FLASH_ATTENTION_AVAILABLE)
    if use_flash and FLASH_ATTENTION_AVAILABLE:
        print(f">>> Replacing {len(model.transformer.h)} attention layers with FlashAttention")
        for i, b in enumerate(model.transformer.h):
            b.attn = FlashSelfAttention(model_cfg)

    # Apply gradient checkpointing if requested
    use_checkpoint = cfg.get("gradient_checkpointing", True)
    if use_checkpoint:
        for i, b in enumerate(model.transformer.h):
            model.transformer.h[i] = CheckpointedBlock(b)

    # Model stays in FP32 - autocast will convert to FP16/BF16 during forward pass
    # This is faster than manual conversion and handles dtype mismatches automatically
    print(f">>> Model in FP32, will use autocast for {precision} training")

    model.config.use_cache = False
    return model