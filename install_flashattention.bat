@echo off

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo FlashAttention Installation
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo Using virtual environment...
echo.

echo Detecting Python and PyTorch versions...
venv\Scripts\python.exe -c "import sys; import torch; print(f'Python: {sys.version_info.major}.{sys.version_info.minor}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"

echo.
echo ========================================
echo Installing FlashAttention 2.7.0.post2
echo ========================================
echo.
echo This pre-built wheel is compatible with:
echo   - Python 3.11
echo   - PyTorch 2.5.1
echo   - CUDA 12.4+ (backwards compatible with your CUDA 13.0)
echo   - Windows 64-bit
echo.
echo Download size: ~182 MB
echo.

venv\Scripts\pip.exe install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl?download=true"

if errorlevel 1 (
    echo.
    echo ========================================
    echo Installation FAILED
    echo ========================================
    echo.
    echo Please check:
    echo 1. Internet connection
    echo 2. Python version is 3.11
    echo 3. PyTorch version is 2.5.x
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Verifying Installation...
echo ========================================
echo.

venv\Scripts\python.exe -c "import flash_attn; print(f'FlashAttention version: {flash_attn.__version__}'); from flash_attn import flash_attn_func; print('Import successful!'); print(''); print('========================================'); print('FlashAttention is WORKING!'); print('========================================')"

if errorlevel 1 (
    echo.
    echo Verification FAILED!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo FlashAttention is now installed and verified.
echo Your training will be 5-10x faster than before!
echo.
echo Next step: Run start.bat to launch the web interface
echo.
pause
