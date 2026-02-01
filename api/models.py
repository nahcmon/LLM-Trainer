from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal


class TrainingConfigRequest(BaseModel):
    # Dataset
    dataset: str = "wikitext/wikitext-2-raw-v1"
    split: str = "train"
    dataset_mixing: str = "interleave"

    # Model Size Preset
    model_size: str = "small-100M"

    # Model Architecture
    vocab_size: int = 32000
    seq_length: int = 1024
    n_layers: int = 12
    hidden_size: int = 768
    n_heads: int = 12
    n_inner: Optional[int] = None

    # Model Hyperparameters
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02

    # Training
    batch_size: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    max_steps: Optional[int] = None

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0

    # Precision
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    gradient_checkpointing: bool = True
    export_format: Literal["safetensors", "gguf", "pytorch"] = "safetensors"

    # Logging
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: Optional[int] = 3


class TrainingStatusResponse(BaseModel):
    state: str
    progress: Dict[str, Any]
    config: Optional[Dict[str, Any]]
    start_time: Optional[str]
    error_message: Optional[str]
