# === AI Hybrid VHS Audio Restorer Installer ===
# Installs: Python (venv), FFmpeg (Portable), PyTorch, VoiceFixer, Demucs
# Fully self-contained: No system modifications, no Admin privileges required.

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=== Setting up Hybrid AI Audio Environment (Portable Mode) ===" -ForegroundColor Cyan

# --- PRE-FLIGHT CHECKS ---

# 1. Check for Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "Found Python: $pyVersion" -ForegroundColor Green
}
catch {
    Write-Error "Python not found in PATH. Please install Python 3.10+ manually and try again."
}



# --- INSTALLATION ---

# 4. Create Virtual Environment
Write-Host "`nStep 1: Setting up Python Environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Created virtual environment."
}

$VenvPy = "$PSScriptRoot\venv\Scripts\python.exe"
$VenvPip = "$PSScriptRoot\venv\Scripts\pip.exe"
$VenvScripts = "$PSScriptRoot\venv\Scripts"

# 4.5 Install FFmpeg Setup (Portable & Enforced Local)
Write-Host "`nStep 1.5: Checking Local FFmpeg..." -ForegroundColor Yellow

# We strictly want FFmpeg inside our venv to ensure self-containment
$localFFmpegPath = "$VenvScripts\ffmpeg.exe"

if (-not (Test-Path $localFFmpegPath)) {
    Write-Host "Local FFmpeg not found. Downloading FULL Portable Build (Gyan.dev)..." -ForegroundColor Cyan
    
    $url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.zip"
    $zip = "$PSScriptRoot\ffmpeg.zip"
    $temp = "$PSScriptRoot\temp_ffmpeg"
    
    # 1. Download
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Write-Host "Downloading (Gyan.dev)..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -UserAgent "Mozilla/5.0"
    }
    catch {
        Write-Warning "Gyan.dev failed. Trying Fallback (GitHub BtbN)..."
        try {
            $urlFallback = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
            Invoke-WebRequest -Uri $urlFallback -OutFile $zip -UseBasicParsing -UserAgent "Mozilla/5.0"
        }
        catch {
            Write-Error "Failed to download FFmpeg from both sources. Please download manually and place 'ffmpeg.exe' in 'venv\Scripts'."
            Write-Error $_.Exception.Message
            exit
        }
    }
    
    # 2. Extract
    Write-Host "Extracting FFmpeg..." -ForegroundColor Cyan
    Expand-Archive -Path $zip -DestinationPath $temp -Force
    
    # 3. Install
    $bin = Get-ChildItem -Path $temp -Recurse -Filter "ffmpeg.exe" | Select-Object -ExpandProperty DirectoryName -First 1
    Copy-Item "$bin\ffmpeg.exe" $VenvScripts -Force
    Copy-Item "$bin\ffprobe.exe" $VenvScripts -Force
    
    # 4. Cleanup
    Remove-Item $zip -Force
    Remove-Item $temp -Recurse -Force
    
    Write-Host "FFmpeg installed to virtual environment." -ForegroundColor Green
}
else {
    Write-Host "Local FFmpeg is is already installed in venv." -ForegroundColor Green
}

# 5. Install Python Dependencies (via requirements.txt)
Write-Host "Step 2: Installing All Dependencies (PyTorch, AI Models, Utilities)..." -ForegroundColor Yellow

try {
    & $VenvPy -m pip install --upgrade pip
    
    # First, install most dependencies from requirements.txt
    # We use a temporary requirements file without the conflicting resemble-enhance
    Write-Host "Installing base dependencies from requirements.txt..." -ForegroundColor Cyan
    $reqContent = Get-Content "$PSScriptRoot\requirements.txt" | Where-Object { $_ -notmatch "resemble-enhance" }
    $tempReq = "$PSScriptRoot\temp_requirements.txt"
    $reqContent | Set-Content $tempReq
    
    & $VenvPip install -r $tempReq --no-cache-dir
    Remove-Item $tempReq
    
    # Then install resemble-enhance with --no-deps to bypass the torch version conflict
    Write-Host "Installing Resemble-Enhance (bypass dependency check)..." -ForegroundColor Cyan
    $resembleUrl = "git+https://github.com/daswer123/resemble-enhance-windows.git"
    & $VenvPip install $resembleUrl --no-deps --no-cache-dir

    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

    Write-Host "All dependencies installed successfully." -ForegroundColor Green
    
    # NEW: Apply Patches (cleaned up)
    Write-Host "Applying runtime patches (DeepSpeed removal + Torchaudio fixes)..." -ForegroundColor Cyan
    & $VenvPy apply_patches.py
    
}
catch {
    Write-Error "An unexpected error occurred during dependency installation."
}

# 7. Create Directories
Write-Host "Step 3: Creating project structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "input" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "temp_work" | Out-Null

# 8. Create a Batch Launcher for the HYBRID Script
Write-Host "Step 4: Creating Launcher..." -ForegroundColor Yellow
$batContent = @"
@echo off
set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" restore_audio_hybrid.py %*
pause
"@
$batContent | Out-File -FilePath "start.bat" -Encoding ascii

Write-Host "`n=== Installation Complete! ===" -ForegroundColor Green
Write-Host "1. Put your video files in the 'input' folder."
Write-Host "2. Double-click 'start.bat' to run the Hybrid AI Cleaner."
Write-Host "Press Enter to exit..."
# Read-Host # Commented out for automation
