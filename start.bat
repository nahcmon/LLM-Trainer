@echo off

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Suppress HuggingFace symlink warnings (cosmetic only, doesn't affect functionality)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo ========================================
echo LLM Training Web App - Starting Server
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Check if web_app.py exists
if not exist "web_app.py" (
    echo ERROR: web_app.py not found
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

echo Verifying installation...
venv\Scripts\python.exe -c "import fastapi" 2>NUL
if errorlevel 1 (
    echo ERROR: FastAPI not installed in virtual environment!
    echo.
    echo Please run setup.bat to install all dependencies.
    echo.
    pause
    exit /b 1
)

echo FastAPI: OK
echo Virtual environment: OK
echo.
echo Starting web server on http://localhost:2345
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Use the venv Python directly (don't rely on activation)
venv\Scripts\python.exe web_app.py

pause
