from typing import Dict, Any

# Parameter categories for UI organization
PARAMETER_CATEGORIES = {
    "dataset": {
        "label": "Dataset Configuration",
        "icon": "📚",
        "parameters": ["dataset", "split", "dataset_mixing"]
    },
    "model_architecture": {
        "label": "Model Architecture",
        "icon": "🏗️",
        "parameters": ["model_size", "vocab_size", "seq_length", "n_layers", "hidden_size", "n_heads", "n_inner"]
    },
    "model_hyperparameters": {
        "label": "Model Hyperparameters",
        "icon": "⚙️",
        "parameters": ["resid_pdrop", "embd_pdrop", "attn_pdrop", "activation_function", "layer_norm_epsilon", "initializer_range"]
    },
    "training": {
        "label": "Training Configuration",
        "icon": "🎯",
        "parameters": ["batch_size", "per_device_train_batch_size", "gradient_accumulation_steps", "epochs", "max_steps"]
    },
    "optimizer": {
        "label": "Optimizer Settings",
        "icon": "📈",
        "parameters": ["learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm"]
    },
    "scheduler": {
        "label": "Learning Rate Scheduler",
        "icon": "📊",
        "parameters": ["lr_scheduler_type", "warmup_steps", "warmup_ratio"]
    },
    "precision": {
        "label": "Precision & Optimization",
        "icon": "⚡",
        "parameters": ["precision", "gradient_checkpointing", "use_flash_attention", "export_format"]
    },
    "logging": {
        "label": "Logging & Checkpointing",
        "icon": "💾",
        "parameters": ["logging_steps", "save_strategy", "save_steps", "save_total_limit"]
    }
}

# Complete parameter definitions with tooltips
PARAMETER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Dataset
    "dataset": {
        "type": "text",
        "default": "wikitext/wikitext-2-raw-v1",
        "label": "Dataset Name or URL",
        "tooltip": "HuggingFace dataset(s) with optional config/subset. For single dataset: 'wikitext/wikitext-2-raw-v1'. For multiple datasets: use comma-separated list like 'wikitext/wikitext-2-raw-v1, openwebtext'. Multiple datasets will be combined using the dataset mixing strategy below.",
        "placeholder": "wikitext/wikitext-2-raw-v1",
        "required": True
    },
    "split": {
        "type": "select",
        "default": "train",
        "label": "Dataset Split",
        "tooltip": "Which split of the dataset to use for training. 'train': standard training data, 'validation': validation/dev set, 'test': test set. Most training uses 'train'.",
        "options": ["train", "validation", "test"],
        "required": True
    },
    "dataset_mixing": {
        "type": "select",
        "default": "interleave",
        "label": "Dataset Mixing Strategy",
        "tooltip": "How to combine multiple datasets (only applies when using multiple datasets). 'interleave': alternates examples from each dataset for balanced mixing. 'concatenate': puts all examples from first dataset, then second, etc. Interleave is recommended for balanced training.",
        "options": ["interleave", "concatenate"],
        "required": True
    },

    # Model Size Preset
    "model_size": {
        "type": "select",
        "default": "small-100M",
        "label": "Model Size Preset",
        "tooltip": "Pre-configured model sizes. Select a preset to automatically set architecture parameters (n_layers, hidden_size, n_heads). Choose 'custom' to manually configure all parameters.",
        "options": [
            "tiny-10M", "tiny-50M", "small-100M", "small-300M",
            "medium-500M", "medium-1B", "large-2B", "large-7B",
            "xl-13B", "xl-30B", "xl-70B", "xxl-175B", "xxl-500B",
            "ultra-1T", "ultra-2T", "custom"
        ],
        "required": True
    },

    # Model Architecture
    "vocab_size": {
        "type": "number",
        "default": 32000,
        "label": "Vocabulary Size",
        "tooltip": "Size of the tokenizer vocabulary. Larger vocabularies can represent more tokens but require more memory. Common values: 32000, 50000.",
        "min": 1000,
        "max": 100000,
        "required": True
    },
    "seq_length": {
        "type": "number",
        "default": 1024,
        "label": "Sequence Length",
        "tooltip": "Maximum number of tokens in a sequence. Longer sequences capture more context but require significantly more VRAM. Powers of 2 are recommended (512, 1024, 2048, 4096).",
        "min": 128,
        "max": 8192,
        "required": True
    },
    "n_layers": {
        "type": "number",
        "default": 12,
        "label": "Number of Layers",
        "tooltip": "Number of transformer layers. More layers increase model capacity and VRAM usage. GPT-2 small uses 12, medium uses 24, large uses 36.",
        "min": 1,
        "max": 96,
        "required": True
    },
    "hidden_size": {
        "type": "number",
        "default": 768,
        "label": "Hidden Size",
        "tooltip": "Dimension of the model's hidden states. Must be divisible by n_heads. Common values: 768 (GPT-2 small), 1024, 1536, 2048.",
        "min": 64,
        "max": 8192,
        "required": True
    },
    "n_heads": {
        "type": "number",
        "default": 12,
        "label": "Number of Attention Heads",
        "tooltip": "Number of attention heads in multi-head attention. hidden_size must be divisible by n_heads. More heads allow the model to attend to different aspects of the input.",
        "min": 1,
        "max": 128,
        "required": True
    },
    "n_inner": {
        "type": "number",
        "default": None,
        "label": "FFN Inner Dimension",
        "tooltip": "Inner dimension of the feed-forward network. If not specified, defaults to 4 * hidden_size. Larger values increase model capacity and VRAM.",
        "min": 64,
        "max": 32768,
        "required": False
    },

    # Model Hyperparameters
    "resid_pdrop": {
        "type": "number",
        "default": 0.1,
        "label": "Residual Dropout",
        "tooltip": "Dropout probability applied to residual connections. Higher values (0.1-0.3) prevent overfitting but may slow convergence. 0.0 disables dropout.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required": True
    },
    "embd_pdrop": {
        "type": "number",
        "default": 0.1,
        "label": "Embedding Dropout",
        "tooltip": "Dropout probability applied to embeddings. Helps prevent overfitting on small datasets. Typical range: 0.0-0.3.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required": True
    },
    "attn_pdrop": {
        "type": "number",
        "default": 0.1,
        "label": "Attention Dropout",
        "tooltip": "Dropout probability applied to attention weights. Helps regularize attention patterns. Typical range: 0.0-0.3.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required": True
    },
    "activation_function": {
        "type": "select",
        "default": "gelu_new",
        "label": "Activation Function",
        "tooltip": "Activation function used in feed-forward networks. 'gelu_new' is standard for GPT models. 'relu' is faster but less expressive.",
        "options": ["gelu", "gelu_new", "relu", "silu", "gelu_fast"],
        "required": True
    },
    "layer_norm_epsilon": {
        "type": "number",
        "default": 1e-5,
        "label": "Layer Norm Epsilon",
        "tooltip": "Small constant added to denominator in layer normalization for numerical stability. Default (1e-5) works well in most cases.",
        "min": 1e-8,
        "max": 1e-3,
        "step": 1e-8,
        "required": True
    },
    "initializer_range": {
        "type": "number",
        "default": 0.02,
        "label": "Initializer Range",
        "tooltip": "Standard deviation of the normal distribution used to initialize weights. GPT-2 uses 0.02. Smaller values (0.01) for larger models.",
        "min": 0.001,
        "max": 0.1,
        "step": 0.001,
        "required": True
    },

    # Training
    "batch_size": {
        "type": "number",
        "default": 1,
        "label": "Batch Size",
        "tooltip": "Number of examples processed together. Larger batches are more stable but require more VRAM. Use gradient_accumulation_steps to simulate larger batches.",
        "min": 1,
        "max": 512,
        "required": True
    },
    "per_device_train_batch_size": {
        "type": "number",
        "default": 1,
        "label": "Per-Device Batch Size",
        "tooltip": "Batch size per GPU/device. Total batch size = per_device_train_batch_size * num_devices * gradient_accumulation_steps.",
        "min": 1,
        "max": 128,
        "required": True
    },
    "gradient_accumulation_steps": {
        "type": "number",
        "default": 1,
        "label": "Gradient Accumulation Steps",
        "tooltip": "Number of steps to accumulate gradients before updating weights. Simulates larger batch sizes without extra VRAM. Effective batch size = batch_size * gradient_accumulation_steps.",
        "min": 1,
        "max": 128,
        "required": True
    },
    "epochs": {
        "type": "number",
        "default": 1,
        "label": "Number of Epochs",
        "tooltip": "Number of complete passes through the training dataset. More epochs may improve performance but risk overfitting. Use max_steps for more control.",
        "min": 1,
        "max": 1000,
        "required": True
    },
    "max_steps": {
        "type": "number",
        "default": None,
        "label": "Max Training Steps",
        "tooltip": "Maximum number of training steps. If specified, overrides epochs. Useful for precise training duration control. Leave empty to use epochs instead.",
        "min": 1,
        "max": 1000000,
        "required": False
    },

    # Optimizer
    "learning_rate": {
        "type": "number",
        "default": 3e-4,
        "label": "Learning Rate",
        "tooltip": "Step size for weight updates. Too high causes instability, too low causes slow learning. Common range: 1e-5 to 1e-3. For AdamW, 3e-4 is a good starting point.",
        "min": 1e-7,
        "max": 1e-2,
        "step": 1e-7,
        "required": True
    },
    "weight_decay": {
        "type": "number",
        "default": 0.01,
        "label": "Weight Decay",
        "tooltip": "L2 regularization coefficient. Helps prevent overfitting by penalizing large weights. Common values: 0.0 (no decay) to 0.1. AdamW typically uses 0.01.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.001,
        "required": True
    },
    "adam_beta1": {
        "type": "number",
        "default": 0.9,
        "label": "Adam Beta 1",
        "tooltip": "Exponential decay rate for first moment estimates in Adam optimizer. Controls momentum. Default (0.9) works well for most cases.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required": True
    },
    "adam_beta2": {
        "type": "number",
        "default": 0.999,
        "label": "Adam Beta 2",
        "tooltip": "Exponential decay rate for second moment estimates in Adam optimizer. Controls adaptive learning rate. Default (0.999) works well for most cases.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.001,
        "required": True
    },
    "adam_epsilon": {
        "type": "number",
        "default": 1e-8,
        "label": "Adam Epsilon",
        "tooltip": "Small constant for numerical stability in Adam optimizer. Prevents division by zero. Default (1e-8) is standard.",
        "min": 1e-10,
        "max": 1e-6,
        "step": 1e-10,
        "required": True
    },
    "max_grad_norm": {
        "type": "number",
        "default": 1.0,
        "label": "Max Gradient Norm",
        "tooltip": "Maximum gradient norm for gradient clipping. Prevents exploding gradients. Values like 1.0 or 0.5 are common. Set to 0 to disable clipping.",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "required": True
    },

    # Scheduler
    "lr_scheduler_type": {
        "type": "select",
        "default": "linear",
        "label": "LR Scheduler Type",
        "tooltip": "Learning rate schedule. 'linear': linear decay from learning_rate to 0. 'cosine': cosine decay. 'constant': no decay. 'constant_with_warmup': constant after warmup.",
        "options": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt"],
        "required": True
    },
    "warmup_steps": {
        "type": "number",
        "default": 0,
        "label": "Warmup Steps",
        "tooltip": "Number of steps to linearly increase learning rate from 0 to learning_rate. Helps stabilize early training. Common: 500-2000 steps or 5-10% of total steps.",
        "min": 0,
        "max": 10000,
        "required": True
    },
    "warmup_ratio": {
        "type": "number",
        "default": 0.0,
        "label": "Warmup Ratio",
        "tooltip": "Alternative to warmup_steps. Fraction of total steps to use for warmup. If specified, overrides warmup_steps. Common: 0.05 to 0.1.",
        "min": 0.0,
        "max": 0.5,
        "step": 0.01,
        "required": True
    },

    # Precision
    "precision": {
        "type": "select",
        "default": "fp16",
        "label": "Training Precision",
        "tooltip": "Numerical precision for training. 'fp16': fastest, uses least VRAM, may have stability issues. 'bf16': better stability than fp16, requires Ampere+ GPUs. 'fp32': most stable, slowest.",
        "options": ["fp16", "bf16", "fp32"],
        "required": True
    },
    "gradient_checkpointing": {
        "type": "checkbox",
        "default": True,
        "label": "Gradient Checkpointing",
        "tooltip": "Trade compute for memory by recomputing activations during backward pass. Reduces VRAM usage by ~50% but increases training time by ~20%. Essential for large models.",
        "required": True
    },
    "use_flash_attention": {
        "type": "checkbox",
        "default": True,
        "label": "Use FlashAttention",
        "tooltip": "Enable FlashAttention for 5-10x faster training. Automatically detected and enabled if installed. Disable only for debugging. Requires FlashAttention to be installed via setup.bat.",
        "required": True
    },

    # Model Export
    "export_format": {
        "type": "select",
        "default": "safetensors",
        "label": "Export Format",
        "tooltip": "Output format for the trained model. 'safetensors': HuggingFace standard, safe & fast. 'gguf': llama.cpp format for CPU inference. 'pytorch': standard PyTorch .bin format.",
        "options": ["safetensors", "gguf", "pytorch"],
        "required": True
    },

    # Logging
    "logging_steps": {
        "type": "number",
        "default": 10,
        "label": "Logging Steps",
        "tooltip": "Log training metrics (loss, learning rate) every N steps. More frequent logging (10-50) helps monitor training but may slow it down slightly.",
        "min": 1,
        "max": 1000,
        "required": True
    },
    "save_strategy": {
        "type": "select",
        "default": "steps",
        "label": "Save Strategy",
        "tooltip": "When to save model checkpoints. 'steps': save every N steps. 'epoch': save at end of each epoch. 'no': don't save checkpoints.",
        "options": ["no", "steps", "epoch"],
        "required": True
    },
    "save_steps": {
        "type": "number",
        "default": 500,
        "label": "Save Steps",
        "tooltip": "Save checkpoint every N steps (when save_strategy='steps'). Balance between safety (frequent saves) and disk space. Common: 500-2000.",
        "min": 1,
        "max": 10000,
        "required": True
    },
    "save_total_limit": {
        "type": "number",
        "default": 3,
        "label": "Max Checkpoints to Keep",
        "tooltip": "Maximum number of checkpoints to keep. Older checkpoints are deleted. Saves disk space. Set to None to keep all checkpoints.",
        "min": 1,
        "max": 100,
        "required": False
    }
}
