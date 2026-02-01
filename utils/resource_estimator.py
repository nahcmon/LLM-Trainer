"""
VRAM and training time estimation utilities.

These are rough estimates based on empirical observations and may not be perfectly accurate.
"""

def estimate_vram_usage(
    vocab_size,
    n_layers,
    hidden_size,
    n_heads,
    seq_length,
    batch_size,
    precision="fp16",
    gradient_checkpointing=True
):
    """
    Estimate VRAM usage in GB for training a transformer model.

    Args:
        vocab_size: Vocabulary size
        n_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        n_heads: Number of attention heads
        seq_length: Maximum sequence length
        batch_size: Training batch size
        precision: One of ["fp32", "fp16", "bf16"]
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        dict with breakdown of VRAM usage
    """
    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2
    }[precision]

    # Calculate number of parameters
    embedding_params = vocab_size * hidden_size + seq_length * hidden_size
    layer_params = 12 * (hidden_size ** 2)
    total_layer_params = n_layers * layer_params
    output_params = vocab_size * hidden_size
    total_params = embedding_params + total_layer_params + output_params

    # Model weights
    model_size_gb = (total_params * bytes_per_param) / (1024 ** 3)

    # Optimizer states (AdamW stores 2 states per parameter)
    # Optimizer states are always in fp32
    optimizer_size_gb = (total_params * 4 * 2) / (1024 ** 3)

    # Gradients (same size as model)
    gradient_size_gb = model_size_gb

    # Activations (this is the tricky part)
    # For transformer: activation size ~= batch_size * seq_length * hidden_size * n_layers * factor
    # factor depends on what's stored (attention, FFN, layer norms)
    # Gradient checkpointing reduces activation memory significantly
    activation_factor = 16 if not gradient_checkpointing else 4
    activation_size_gb = (
        batch_size * seq_length * hidden_size * n_layers * activation_factor * bytes_per_param
    ) / (1024 ** 3)

    # Temporary buffers and overhead (~20% of total)
    base_total = model_size_gb + optimizer_size_gb + gradient_size_gb + activation_size_gb
    overhead_gb = base_total * 0.2

    # Total VRAM
    total_vram_gb = base_total + overhead_gb

    return {
        "model": round(model_size_gb, 2),
        "optimizer": round(optimizer_size_gb, 2),
        "gradients": round(gradient_size_gb, 2),
        "activations": round(activation_size_gb, 2),
        "overhead": round(overhead_gb, 2),
        "total": round(total_vram_gb, 2)
    }


def estimate_training_time(
    dataset_size,
    epochs,
    batch_size,
    max_steps=None,
    model_params_millions=100,
    gpu_type="unknown"
):
    """
    Estimate training time.

    This is a very rough estimate based on typical training speeds.

    Args:
        dataset_size: Number of training examples
        epochs: Number of epochs
        batch_size: Batch size
        max_steps: Maximum steps (overrides epochs if set)
        model_params_millions: Model size in millions of parameters
        gpu_type: Type of GPU (for better estimates)

    Returns:
        dict with time estimates
    """
    # Calculate total steps
    if max_steps:
        total_steps = max_steps
    else:
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * epochs

    # Estimate steps per second based on model size and GPU
    # These are rough estimates for typical GPUs
    gpu_speed_multipliers = {
        "RTX 4090": 1.5,
        "RTX 4080": 1.3,
        "RTX 3090": 1.2,
        "RTX 3080": 1.0,
        "V100": 1.0,
        "A100": 2.0,
        "H100": 3.0,
        "unknown": 1.0
    }

    multiplier = gpu_speed_multipliers.get(gpu_type, 1.0)

    # Base speed: larger models are slower
    # Very rough approximation: ~10 steps/sec for 100M params on RTX 3080
    if model_params_millions < 100:
        base_speed = 15.0
    elif model_params_millions < 500:
        base_speed = 10.0
    elif model_params_millions < 1000:
        base_speed = 5.0
    elif model_params_millions < 5000:
        base_speed = 2.0
    else:
        base_speed = 0.5

    steps_per_second = base_speed * multiplier

    # Calculate time
    total_seconds = total_steps / steps_per_second

    # Format time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return {
        "total_seconds": round(total_seconds, 1),
        "total_steps": total_steps,
        "estimated_speed_steps_per_sec": round(steps_per_second, 2),
        "formatted": f"{hours}h {minutes}m {seconds}s",
        "note": "This is a rough estimate. Actual time may vary significantly."
    }


def format_vram_estimate(vram_dict):
    """Format VRAM estimate for display."""
    lines = [
        f"Model: {vram_dict['model']}GB",
        f"Optimizer: {vram_dict['optimizer']}GB",
        f"Gradients: {vram_dict['gradients']}GB",
        f"Activations: {vram_dict['activations']}GB",
        f"Overhead: {vram_dict['overhead']}GB",
        f"────────────────",
        f"Total: {vram_dict['total']}GB"
    ]
    return "\n".join(lines)
