# LLM Training Web Interface

A comprehensive web-based interface for training GPT-style language models from scratch with real-time monitoring and configuration.

## Quick Start

### 1. Setup (First Time Only)

Run the setup script to install all dependencies:

```bash
setup.bat
```

This will:
- Create a virtual environment
- Install all required packages (PyTorch, Transformers, FastAPI, etc.)
- Install FlashAttention-2 for 5-10x faster training
- Set up the web application

### 2. Start the Web App

```bash
start.bat
```

Then open your browser to: **http://localhost:2345**

## Features

### 🎨 **Comprehensive Configuration**
- **40+ training parameters** organized into 8 intuitive categories
- **Interactive tooltips** (?) explaining every parameter with typical values
- **Model presets** from 10M to 70B parameters (GPT-2, GPT-3, LLaMA sizes)
- **Real-time VRAM estimation** as you adjust parameters
- **Multiple dataset support** with interleaving or concatenation
- **Advanced optimizer settings** (AdamW with β1, β2, ε configuration)
- **Learning rate schedulers** (linear, cosine, constant with warmup)
- **Precision options** (FP16, BF16, FP32) with gradient checkpointing

### 📊 **Live Monitoring**
- **Real-time loss tracking** with step-by-step updates
- **Training progress bar** showing completion percentage
- **GPU metrics** (utilization, memory, temperature)
- **Elapsed time** and estimated time remaining
- **Color-coded log streaming** with auto-scroll
- **WebSocket-based updates** for instant feedback (no polling)

### 🎛️ **Easy Control**
- **One-click training** start/stop controls
- **Auto-saving checkpoints** at configurable intervals
- **Resume from checkpoints** to continue interrupted training
- **Background training** - close browser, training continues
- **Graceful shutdown** with checkpoint saving on stop

### 🔧 **Advanced Features**
- **FlashAttention-2 support** for 5-10x faster attention computation
- **Gradient checkpointing** to train larger models with less VRAM
- **Gradient accumulation** to simulate large batch sizes
- **Gradient clipping** to prevent training instability
- **Mixed precision training** with automatic loss scaling
- **Multi-dataset training** - combine datasets with custom mixing
- **Model export** to SafeTensors, GGUF (llama.cpp), or PyTorch formats
- **SentencePiece tokenizer** trained on your dataset
- **Automatic fallbacks** (SDPA → manual attention if FlashAttention unavailable)

## Web Interface Overview

### Left Panel: Configuration (8 Categories)
- **📚 Dataset** - Choose HuggingFace datasets, combine multiple datasets, configure mixing
- **🏗️ Model Architecture** - Layers, hidden size, attention heads, FFN inner dimension
- **⚙️ Model Hyperparameters** - Dropout rates, activation functions, layer norm epsilon
- **🎯 Training** - Batch size, epochs, gradient accumulation, max steps
- **📈 Optimizer** - Learning rate, weight decay, AdamW beta parameters, epsilon
- **📊 Scheduler** - LR scheduling (linear/cosine/constant), warmup steps, decay
- **⚡ Precision** - FP16/BF16/FP32, gradient checkpointing, gradient clipping
- **💾 Logging** - Checkpoint intervals, logging frequency, output directory

### Right Panel: Monitoring
- **Progress Stats** - Current step, epoch, loss, learning rate, GPU stats
- **Progress Bar** - Visual training completion with percentage
- **Training Logs** - Real-time color-coded log streaming with auto-scroll
- **VRAM Estimate** - Before training starts, see estimated memory usage

## Configuration Tips

### Quick Training Test (< 1 minute)
1. Select model preset: "tiny-10M" (10 million parameters)
2. Set `epochs` to 1
3. Set `logging_steps` to 10
4. Keep default dataset (WikiText)
5. Click "Start Training"

### Memory-Efficient Settings (for 8GB VRAM)
- Enable `gradient_checkpointing` ✓
- Use `fp16` precision
- Use gradient accumulation (set `gradient_accumulation_steps` to 4)
- Keep `seq_length` ≤ 1024
- Choose model preset ≤ 500M parameters

### Performance Optimization (for speed)
- Install FlashAttention-2 (automatic in setup.bat)
- Use `fp16` or `bf16` precision (2x faster than fp32)
- Disable gradient checkpointing if VRAM allows
- Increase `batch_size` or `gradient_accumulation_steps`
- Close other GPU applications

### Model Size Selection
Use the **Model Preset** dropdown:
- **tiny-10M / tiny-50M**: Testing and debugging (< 1GB VRAM)
- **small-100M**: Quick experiments, similar to GPT-2 Small (~3GB VRAM)
- **medium-350M**: GPT-2 Medium size (~5GB VRAM)
- **medium-500M**: Balanced quality/speed (~7GB VRAM)
- **large-1B+**: High quality models (12GB+ VRAM required)

### Hover Over "?" Icons
Every parameter has a tooltip explaining:
- What it does
- Typical values
- When to adjust it
- Impact on training and memory

## Troubleshooting

### Server Issues

**"Server stopped/crashed immediately"**
- This is NORMAL! The server is actually running.
- When you see "Application startup complete", the server is ready.
- The command window stays open waiting for requests - don't close it!
- Open your browser to http://localhost:2345 to see the interface.
- To verify: run `test_server.bat` in another terminal

**Port Already in Use**
- Change the port in `web_app.py` (line with `port=2345`)
- Or close the other application using port 2345

**"pynvml deprecated" warning**
- This warning is harmless and will be gone after running the new `setup.bat`
- The new version uses `nvidia-ml-py` instead

### GPU/CUDA Issues

**"PyTorch CPU version detected"**
- Re-run `setup.bat` and choose option 1 (GPU/CUDA)
- Make sure you have NVIDIA GPU drivers installed
- Run `nvidia-smi` in command prompt to verify GPU is detected

**CUDA Out of Memory**
- Reduce `batch_size`
- Enable `gradient_checkpointing`
- Reduce `seq_length`
- Reduce `hidden_size` or `n_layers`

**FlashAttention Not Available**
- The trainer will automatically fall back to standard attention
- Performance may be slower but training will work
- To install FlashAttention: re-run `setup.bat` and choose GPU option
- Requires: NVIDIA GPU, CUDA drivers, Visual Studio Build Tools

### Testing Your Installation

Run this to verify everything is working:
```bash
test_server.bat
```

This will check:
- ✓ Python environment
- ✓ PyTorch and CUDA
- ✓ FlashAttention availability
- ✓ Server connectivity

## File Structure

```
llm_trainer/
├── web_app.py                     # FastAPI application entry point (port 2345)
├── main.py                        # CLI entry point (legacy)
├── trainer.py                     # Core training loop with callbacks
├── model.py                       # GPT-2 with FlashAttention/SDPA
├── model_export.py                # Export to SafeTensors/GGUF/PyTorch
├── dataset.py                     # HuggingFace dataset loader (multi-dataset support)
├── tokenizer.py                   # SentencePiece tokenizer training
├── gpu_utils.py                   # GPU monitoring and VRAM estimation
├── setup.bat                      # One-time setup script
├── start.bat                      # Start the web server
├── requirements.txt               # Python dependencies
├── api/                           # FastAPI backend
│   ├── routes.py                  # REST API endpoints
│   ├── websocket.py               # WebSocket handler for real-time updates
│   └── models.py                  # Pydantic models for API
├── core/                          # Core training logic
│   └── training_manager.py        # Singleton managing training state
├── utils/                         # Utilities
│   ├── parameter_definitions.py   # 40+ parameter definitions with tooltips
│   ├── model_presets.py           # Model size presets (10M - 70B)
│   └── resource_estimator.py      # VRAM and compute estimation
├── static/                        # Frontend (vanilla JS)
│   ├── index.html                 # Main UI
│   ├── css/styles.css             # Styling
│   └── js/
│       ├── main.js                # UI initialization
│       ├── config-form.js         # Dynamic form generation
│       ├── websocket-client.js    # WebSocket client
│       ├── tooltips.js            # Tooltip system
│       ├── model-presets.js       # Model preset dropdown
│       └── resource-estimator.js  # VRAM estimation UI
└── output/                        # Training outputs (created on first run)
    ├── checkpoints/               # Model checkpoints
    └── logs/                      # Training logs
```

## Technical Stack

**Backend:**
- PyTorch 2.1+ with CUDA 12.1
- Transformers (HuggingFace) for GPT-2 architecture
- FastAPI + Uvicorn for web server
- WebSockets for real-time updates
- SentencePiece for tokenization

**Frontend:**
- Vanilla JavaScript (no frameworks)
- WebSocket client for live updates
- Responsive CSS with dark theme

**Optimization:**
- FlashAttention-2 (5-10x faster attention)
- Gradient checkpointing (2-3x VRAM reduction)
- Mixed precision training (FP16/BF16)
- PyTorch SDPA fallback (if FlashAttention unavailable)

## CLI Usage (Legacy)

If you prefer the command line:

```bash
python main.py
```

Edit `main.py` to configure training parameters. The CLI version uses the same trainer but without the web interface.

## System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 10GB disk space for dependencies
- NVIDIA GPU with 4GB+ VRAM (or CPU, much slower)

**Recommended:**
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- CUDA 11.8+ drivers

**Optimal:**
- Python 3.11
- 32GB+ RAM
- NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100)
- CUDA 12.1+ drivers

## Export Formats

After training, export your model in multiple formats:

- **SafeTensors** (default) - HuggingFace standard, fastest loading
- **GGUF** - For llama.cpp inference (CPU/GPU)
- **PyTorch** - Standard .pth checkpoint

Configure export format in the web UI or via `export_format` parameter.

## License

MIT License - Feel free to use and modify!
