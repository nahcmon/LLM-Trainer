@echo off
echo ========================================
echo Visual Studio Build Tools Installer
echo ========================================
echo.
echo This will install the C++ compiler needed for FlashAttention.
echo.
echo IMPORTANT: This requires Administrator privileges!
echo.
echo The script will:
echo 1. Download VS Build Tools (~3 MB)
echo 2. Install C++ Build Tools (~6 GB)
echo 3. Take 10-20 minutes
echo.
pause

powershell -ExecutionPolicy Bypass -File "%~dp0install_build_tools.ps1"

pause
