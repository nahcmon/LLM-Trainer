# PowerShell script to install Visual Studio Build Tools
# Run this as Administrator: Right-click -> Run as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visual Studio Build Tools Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as administrator:" -ForegroundColor Yellow
    Write-Host "1. Right-click on this file (install_build_tools.ps1)" -ForegroundColor Yellow
    Write-Host "2. Select 'Run with PowerShell'" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Detected Administrator privileges. Continuing..." -ForegroundColor Green
Write-Host ""

# Check if Visual Studio Build Tools already installed
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    Write-Host "Checking for existing Visual Studio installations..." -ForegroundColor Yellow
    $buildTools = & $vsWhere -products * -requires Microsoft.VisualStudio.Workload.VCTools -property installationPath

    if ($buildTools) {
        Write-Host ""
        Write-Host "Visual Studio Build Tools already installed at:" -ForegroundColor Green
        Write-Host $buildTools -ForegroundColor Green
        Write-Host ""
        $continue = Read-Host "Do you want to reinstall/update? (Y/N)"
        if ($continue -ne "Y" -and $continue -ne "y") {
            Write-Host "Installation cancelled." -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 0
        }
    }
}

Write-Host "Downloading Visual Studio Build Tools installer..." -ForegroundColor Cyan
$installerPath = "$env:TEMP\vs_buildtools.exe"

try {
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile $installerPath
    Write-Host "Download complete!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to download installer!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing Build Tools..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will install:" -ForegroundColor Yellow
Write-Host "  - C++ Build Tools" -ForegroundColor Yellow
Write-Host "  - MSVC C++ Compiler" -ForegroundColor Yellow
Write-Host "  - Windows SDK" -ForegroundColor Yellow
Write-Host ""
Write-Host "Installation size: ~6 GB" -ForegroundColor Yellow
Write-Host "Installation time: 10-20 minutes" -ForegroundColor Yellow
Write-Host ""

$continue = Read-Host "Continue with installation? (Y/N)"
if ($continue -ne "Y" -and $continue -ne "y") {
    Write-Host "Installation cancelled." -ForegroundColor Yellow
    Remove-Item $installerPath -Force
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "Starting installation (this will take 10-20 minutes)..." -ForegroundColor Cyan
Write-Host "A separate installer window will open. Please wait..." -ForegroundColor Yellow
Write-Host ""

# Install with C++ workload
$installArgs = @(
    "--quiet",
    "--wait",
    "--norestart",
    "--nocache",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    "--add", "Microsoft.VisualStudio.Component.Windows11SDK.22000"
)

try {
    $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru

    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "Installation Complete!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Visual Studio Build Tools with C++ have been installed." -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: You may need to restart your computer" -ForegroundColor Yellow
        Write-Host "for the changes to take effect." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "After restart, run: setup.bat" -ForegroundColor Cyan
        Write-Host "And choose to install FlashAttention" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "Installation completed with exit code: $($process.ExitCode)" -ForegroundColor Yellow
        Write-Host "This may indicate a partial installation." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Try running setup.bat anyway to see if it works." -ForegroundColor Cyan
    }
} catch {
    Write-Host ""
    Write-Host "ERROR during installation!" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
} finally {
    # Cleanup
    if (Test-Path $installerPath) {
        Remove-Item $installerPath -Force
    }
}

Write-Host ""
Read-Host "Press Enter to exit"
