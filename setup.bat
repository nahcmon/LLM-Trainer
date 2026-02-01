@echo off
cd /d "%~dp0"

echo ========================================
echo LLM Training Web App - Setup (GPU)
echo ========================================
echo Auto-configuring for GPU (CUDA) training...
echo.

python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    exit /b 1
)

echo.
echo Creating virtual environment...
if exist venv (
    echo Removing old venv...
    rmdir /s /q venv
    if errorlevel 1 (
        echo ERROR: Could not remove old venv
        exit /b 1
    )
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv
    exit /b 1
)

echo.
echo Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed!
    exit /b 1
)

echo.
echo ========================================
echo Installing PyTorch with CUDA 12.1...
echo ========================================
echo This is optimized for your RTX 3080 Ti
echo Download size: ~2.4 GB, please be patient...
echo.

venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: PyTorch installation failed!
    exit /b 1
)

echo.
echo Installing ML dependencies...
venv\Scripts\pip.exe install transformers>=4.36.0 datasets>=2.16.0 sentencepiece tqdm nvidia-ml-py>=12.535.0 packaging wheel safetensors>=0.4.0 gguf>=0.6.0
if errorlevel 1 (
    echo ERROR: ML dependencies failed!
    exit /b 1
)

echo.
echo Installing web dependencies...
venv\Scripts\pip.exe install fastapi>=0.104.0 "uvicorn[standard]>=0.24.0" python-multipart>=0.0.6 websockets>=12.0 pydantic>=2.0.0
if errorlevel 1 (
    echo ERROR: Web dependencies failed!
    exit /b 1
)

echo.
echo ========================================
echo Installing FlashAttention...
echo ========================================
echo This will make training 5-10x faster on your GPU!
echo.

venv\Scripts\pip.exe install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl?download=true"
if errorlevel 1 (
    echo.
    echo WARNING: FlashAttention installation failed.
    echo Training will still work, just 10-20%% slower.
    echo.
) else (
    echo FlashAttention installed successfully!
)

echo.
echo ========================================
echo Verifying installation...
echo ========================================
echo.

venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
venv\Scripts\python.exe -c "import transformers; print(f'Transformers: {transformers.__version__}')"
venv\Scripts\python.exe -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
venv\Scripts\python.exe -c "import gguf; print('GGUF: Installed')"
venv\Scripts\python.exe -c "import safetensors; print(f'SafeTensors: {safetensors.__version__}')"
venv\Scripts\python.exe -c "import importlib.util; fa = importlib.util.find_spec('flash_attn'); print(f'FlashAttention: ENABLED') if fa else print('FlashAttention: Not installed (using PyTorch SDPA)')"

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Your RTX 3080 Ti is ready for training!
echo You can now run start.bat to launch the web app.
echo.
