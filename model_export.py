import torch
import os
from pathlib import Path

def export_model(model, tokenizer_path, output_dir="output", export_format="safetensors"):
    """
    Export trained model in the specified format.

    Args:
        model: Trained PyTorch model
        tokenizer_path: Path to tokenizer model file
        output_dir: Directory to save exported model
        export_format: One of ['safetensors', 'gguf', 'pytorch']

    Returns:
        str: Path to exported model file
    """
    os.makedirs(output_dir, exist_ok=True)

    if export_format == "safetensors":
        return _export_safetensors(model, tokenizer_path, output_dir)
    elif export_format == "gguf":
        try:
            return _export_gguf(model, tokenizer_path, output_dir)
        except ImportError as e:
            print(f"Warning: {e}")
            print("Falling back to safetensors format...")
            return _export_safetensors(model, tokenizer_path, output_dir)
    elif export_format == "pytorch":
        return _export_pytorch(model, tokenizer_path, output_dir)
    else:
        raise ValueError(f"Unknown export format: {export_format}")


def _export_safetensors(model, tokenizer_path, output_dir):
    """Export model in safetensors format (HuggingFace standard)"""
    from safetensors.torch import save_file
    import shutil

    output_path = Path(output_dir)

    # Save model state dict as safetensors
    model_file = output_path / "model.safetensors"
    save_file(model.state_dict(), str(model_file))

    # Save config
    config_file = output_path / "config.json"
    model.config.save_pretrained(str(output_path))

    # Copy tokenizer
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, output_path / "tokenizer.model")
        if os.path.exists(tokenizer_path.replace('.model', '.vocab')):
            shutil.copy(tokenizer_path.replace('.model', '.vocab'),
                       output_path / "tokenizer.vocab")

    return str(model_file)


def _export_pytorch(model, tokenizer_path, output_dir):
    """Export model in PyTorch .bin format"""
    import shutil

    output_path = Path(output_dir)

    # Save model state dict as PyTorch bin
    model_file = output_path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_file)

    # Save config
    model.config.save_pretrained(str(output_path))

    # Copy tokenizer
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, output_path / "tokenizer.model")
        if os.path.exists(tokenizer_path.replace('.model', '.vocab')):
            shutil.copy(tokenizer_path.replace('.model', '.vocab'),
                       output_path / "tokenizer.vocab")

    return str(model_file)


def _export_gguf(model, tokenizer_path, output_dir):
    """
    Export model in GGUF format for llama.cpp.

    Note: This is a simplified implementation. Full GGUF export requires
    additional metadata and tensor quantization.
    """
    try:
        import gguf
        import numpy as np
    except ImportError:
        raise ImportError(
            "gguf package not installed. Install with: pip install gguf\n"
            "Note: GGUF export is experimental and may not work for all models."
        )

    output_path = Path(output_dir)
    model_file = output_path / "model.gguf"

    # Create GGUF writer
    writer = gguf.GGUFWriter(str(model_file), arch="gpt2")

    # Add metadata
    writer.add_name("LLM Trainer Model")
    writer.add_architecture()
    writer.add_context_length(model.config.n_ctx)
    writer.add_embedding_length(model.config.n_embd)
    writer.add_block_count(model.config.n_layer)
    writer.add_head_count(model.config.n_head)

    # Convert and add tensors
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        # Convert to numpy and add to GGUF
        np_tensor = tensor.cpu().float().numpy()
        writer.add_tensor(name, np_tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return str(model_file)
