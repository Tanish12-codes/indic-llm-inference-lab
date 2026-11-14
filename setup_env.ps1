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

# PyTorch with CUDA 12.4 wheels. The RTX 3050 works with any recent NVIDIA driver (>= 550).
# If this index fails, try cu121 or cu126 instead.
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124

& $py -m pip install -r requirements.txt

Write-Host ""
Write-Host "Running environment check ..."
& $py scripts\env_check.py
