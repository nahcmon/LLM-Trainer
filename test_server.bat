@echo off

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo Testing LLM Training Web Server
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo Using virtual environment Python...
echo.

echo Testing Python environment...
venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo Testing FlashAttention...
venv\Scripts\python.exe -c "try: import flash_attn; print('FlashAttention: AVAILABLE'); except: print('FlashAttention: Not installed (standard attention will be used)')"

echo.
echo Testing FastAPI imports...
venv\Scripts\python.exe -c "from fastapi import FastAPI; from api.routes import router; print('FastAPI: OK')"

echo.
echo Testing if server is running...
echo Attempting to connect to http://localhost:2345
echo.

venv\Scripts\python.exe -c "import urllib.request; import json; response = urllib.request.urlopen('http://localhost:2345/api/parameters', timeout=5); data = json.loads(response.read()); print(f'Server is RUNNING! Found {len(data[\"definitions\"])} parameters configured.')" 2>NUL

if errorlevel 1 (
    echo.
    echo Server is NOT running or not responding.
    echo Please start it with: start.bat
) else (
    echo.
    echo ========================================
    echo All tests passed! Server is working!
    echo ========================================
    echo.
    echo Open your browser to: http://localhost:2345
)

echo.
pause
