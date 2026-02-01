"""
Model size presets for different parameter counts.

These presets are based on scaling laws and common architectures like GPT-2, GPT-3, LLaMA, etc.
"""

# Model size presets: name -> (n_layers, hidden_size, n_heads, n_inner)
MODEL_SIZE_PRESETS = {
    "tiny-10M": {
        "description": "~10M parameters - Testing/debugging",
        "n_layers": 6,
        "hidden_size": 384,
        "n_heads": 6,
        "n_inner": 1536,
        "params": 10_000_000
    },
    "tiny-50M": {
        "description": "~50M parameters - Small experiments",
        "n_layers": 8,
        "hidden_size": 512,
        "n_heads": 8,
        "n_inner": 2048,
        "params": 50_000_000
    },
    "small-100M": {
        "description": "~100M parameters - Similar to GPT-2 Small",
        "n_layers": 12,
        "hidden_size": 768,
        "n_heads": 12,
        "n_inner": 3072,
        "params": 100_000_000
    },
    "small-150M": {
        "description": "~150M parameters",
        "n_layers": 16,
        "hidden_size": 896,
        "n_heads": 14,
        "n_inner": 3584,
        "params": 150_000_000
    },
    "small-200M": {
        "description": "~200M parameters",
        "n_layers": 20,
        "hidden_size": 960,
        "n_heads": 15,
        "n_inner": 3840,
        "params": 200_000_000
    },
    "small-250M": {
        "description": "~250M parameters",
        "n_layers": 22,
        "hidden_size": 1024,
        "n_heads": 16,
        "n_inner": 4096,
        "params": 250_000_000
    },
    "small-300M": {
        "description": "~300M parameters - Similar to GPT-2 Medium",
        "n_layers": 24,
        "hidden_size": 1024,
        "n_heads": 16,
        "n_inner": 4096,
        "params": 300_000_000
    },
    "small-350M": {
        "description": "~350M parameters",
        "n_layers": 24,
        "hidden_size": 1152,
        "n_heads": 18,
        "n_inner": 4608,
        "params": 350_000_000
    },
    "small-400M": {
        "description": "~400M parameters",
        "n_layers": 28,
        "hidden_size": 1152,
        "n_heads": 18,
        "n_inner": 4608,
        "params": 400_000_000
    },
    "small-450M": {
        "description": "~450M parameters",
        "n_layers": 24,
        "hidden_size": 1216,
        "n_heads": 19,
        "n_inner": 4864,
        "params": 450_000_000
    },
    "medium-500M": {
        "description": "~500M parameters - Educational/research",
        "n_layers": 24,
        "hidden_size": 1280,
        "n_heads": 20,
        "n_inner": 5120,
        "params": 500_000_000
    },
    "medium-550M": {
        "description": "~550M parameters",
        "n_layers": 26,
        "hidden_size": 1280,
        "n_heads": 20,
        "n_inner": 5120,
        "params": 550_000_000
    },
    "medium-600M": {
        "description": "~600M parameters",
        "n_layers": 28,
        "hidden_size": 1280,
        "n_heads": 20,
        "n_inner": 5120,
        "params": 600_000_000
    },
    "medium-650M": {
        "description": "~650M parameters",
        "n_layers": 30,
        "hidden_size": 1280,
        "n_heads": 20,
        "n_inner": 5120,
        "params": 650_000_000
    },
    "medium-700M": {
        "description": "~700M parameters",
        "n_layers": 28,
        "hidden_size": 1344,
        "n_heads": 21,
        "n_inner": 5376,
        "params": 700_000_000
    },
    "medium-750M": {
        "description": "~750M parameters",
        "n_layers": 30,
        "hidden_size": 1344,
        "n_heads": 21,
        "n_inner": 5376,
        "params": 750_000_000
    },
    "medium-800M": {
        "description": "~800M parameters",
        "n_layers": 32,
        "hidden_size": 1344,
        "n_heads": 21,
        "n_inner": 5376,
        "params": 800_000_000
    },
    "medium-850M": {
        "description": "~850M parameters",
        "n_layers": 32,
        "hidden_size": 1408,
        "n_heads": 22,
        "n_inner": 5632,
        "params": 850_000_000
    },
    "medium-900M": {
        "description": "~900M parameters",
        "n_layers": 34,
        "hidden_size": 1408,
        "n_heads": 22,
        "n_inner": 5632,
        "params": 900_000_000
    },
    "medium-950M": {
        "description": "~950M parameters",
        "n_layers": 34,
        "hidden_size": 1472,
        "n_heads": 23,
        "n_inner": 5888,
        "params": 950_000_000
    },
    "medium-1B": {
        "description": "~1B parameters - Similar to GPT-2 Large",
        "n_layers": 36,
        "hidden_size": 1536,
        "n_heads": 24,
        "n_inner": 6144,
        "params": 1_000_000_000
    },
    "large-1.5B": {
        "description": "~1.5B parameters",
        "n_layers": 32,
        "hidden_size": 1792,
        "n_heads": 28,
        "n_inner": 7168,
        "params": 1_500_000_000
    },
    "large-2B": {
        "description": "~2B parameters - Serious training",
        "n_layers": 32,
        "hidden_size": 2048,
        "n_heads": 32,
        "n_inner": 8192,
        "params": 2_000_000_000
    },
    "large-2.5B": {
        "description": "~2.5B parameters",
        "n_layers": 36,
        "hidden_size": 2048,
        "n_heads": 32,
        "n_inner": 8192,
        "params": 2_500_000_000
    },
    "large-3B": {
        "description": "~3B parameters",
        "n_layers": 40,
        "hidden_size": 2048,
        "n_heads": 32,
        "n_inner": 8192,
        "params": 3_000_000_000
    },
    "large-3.5B": {
        "description": "~3.5B parameters",
        "n_layers": 44,
        "hidden_size": 2048,
        "n_heads": 32,
        "n_inner": 8192,
        "params": 3_500_000_000
    },
    "large-4B": {
        "description": "~4B parameters",
        "n_layers": 48,
        "hidden_size": 2048,
        "n_heads": 32,
        "n_inner": 8192,
        "params": 4_000_000_000
    },
    "large-7B": {
        "description": "~7B parameters - Similar to LLaMA 7B",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "n_inner": 11008,
        "params": 7_000_000_000
    },
    "xl-13B": {
        "description": "~13B parameters - Similar to LLaMA 13B",
        "n_layers": 40,
        "hidden_size": 5120,
        "n_heads": 40,
        "n_inner": 13824,
        "params": 13_000_000_000
    },
    "xl-30B": {
        "description": "~30B parameters - Large scale training",
        "n_layers": 60,
        "hidden_size": 6656,
        "n_heads": 52,
        "n_inner": 17920,
        "params": 30_000_000_000
    },
    "xl-70B": {
        "description": "~70B parameters - Similar to LLaMA 70B",
        "n_layers": 80,
        "hidden_size": 8192,
        "n_heads": 64,
        "n_inner": 28672,
        "params": 70_000_000_000
    },
    "xxl-175B": {
        "description": "~175B parameters - Similar to GPT-3",
        "n_layers": 96,
        "hidden_size": 12288,
        "n_heads": 96,
        "n_inner": 49152,
        "params": 175_000_000_000
    },
    "xxl-500B": {
        "description": "~500B parameters - Mega scale",
        "n_layers": 128,
        "hidden_size": 16384,
        "n_heads": 128,
        "n_inner": 65536,
        "params": 500_000_000_000
    },
    "ultra-1T": {
        "description": "~1T parameters - Ultra scale",
        "n_layers": 160,
        "hidden_size": 20480,
        "n_heads": 160,
        "n_inner": 81920,
        "params": 1_000_000_000_000
    },
    "ultra-2T": {
        "description": "~2T parameters - Maximum scale",
        "n_layers": 200,
        "hidden_size": 25600,
        "n_heads": 200,
        "n_inner": 102400,
        "params": 2_000_000_000_000
    },
    "custom": {
        "description": "Custom configuration - Set manually",
        "n_layers": 12,
        "hidden_size": 768,
        "n_heads": 12,
        "n_inner": 3072,
        "params": None
    }
}


def get_model_preset(preset_name: str) -> dict:
    """Get model architecture parameters for a preset size."""
    if preset_name not in MODEL_SIZE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(MODEL_SIZE_PRESETS.keys())}")

    return MODEL_SIZE_PRESETS[preset_name].copy()


def calculate_model_params(vocab_size, n_layers, hidden_size, n_heads, seq_length):
    """
    Calculate approximate number of parameters for a transformer model.

    Formula based on GPT-2/GPT-3 architecture:
    - Embedding: vocab_size * hidden_size + seq_length * hidden_size
    - Each layer: 4 * hidden_size^2 (attention) + 8 * hidden_size^2 (FFN) = 12 * hidden_size^2
    - Output layer: vocab_size * hidden_size
    """
    # Token embeddings + position embeddings
    embedding_params = vocab_size * hidden_size + seq_length * hidden_size

    # Transformer layers (attention + FFN + layer norms)
    # Attention: 4 * hidden_size^2 (Q, K, V, O projections)
    # FFN: 8 * hidden_size^2 (2 linear layers with 4x expansion)
    # Layer norms: ~4 * hidden_size (negligible)
    layer_params = 12 * (hidden_size ** 2)
    total_layer_params = n_layers * layer_params

    # Output layer (LM head)
    output_params = vocab_size * hidden_size

    total = embedding_params + total_layer_params + output_params

    return total


def format_param_count(param_count):
    """Format parameter count in human-readable form."""
    if param_count >= 1_000_000_000_000:
        return f"{param_count / 1_000_000_000_000:.2f}T"
    elif param_count >= 1_000_000_000:
        return f"{param_count / 1_000_000_000:.2f}B"
    elif param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.2f}M"
    elif param_count >= 1_000:
        return f"{param_count / 1_000:.2f}K"
    else:
        return str(param_count)
