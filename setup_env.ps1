# One-time setup for Windows PowerShell.
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# If you already have a .venv, it is reused.

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { python -m venv .venv }
}

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

# PyTorch with CUDA 13.2 wheels (default CUDA build for torch 2.15; supports Python 3.14).
# Needs a recent NVIDIA driver (check: nvidia-smi shows "CUDA Version: 13.2" or higher). Alternative: cu134.
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu132

& $py -m pip install -r requirements.txt

Write-Host ""
Write-Host "Running environment check ..."
& $py scripts\env_check.py
