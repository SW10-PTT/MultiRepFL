# ============================================
# Project setup script (Windows)
# ============================================
#
# Usage:
#   .\windows-setup.ps1 [cpu|nvidia|nvidia-legacy|amd-windows]
#
#   nvidia         — CUDA 13.0 (driver 575+)
#   nvidia-legacy  — CUDA 12.8 (driver 550-574, e.g. driver 570)
#
# NOTE: If you get "running scripts is disabled", run once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# Linux / macOS users: use setup.sh instead.

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("cpu", "nvidia", "nvidia-legacy", "amd-windows")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host "Setting up project ($Mode)"
Write-Host "============================================"

# --------------------------------------------
# Ensure we're in repo root
# --------------------------------------------
Set-Location $PSScriptRoot

# --------------------------------------------
# Git cleanup
# --------------------------------------------
Write-Host ""
Write-Host "==> Resetting git state"

git checkout main
git fetch origin
git reset --hard origin/main

# --------------------------------------------
# Detect required Python version from pyproject.toml
# --------------------------------------------
Write-Host ""
Write-Host "==> Detecting required Python version"

if (-not (Test-Path "pyproject.toml")) {
    Write-Error "ERROR: pyproject.toml not found"
    exit 1
}

$REQUIRED_PYTHON = @"
import re
from pathlib import Path

text = Path("pyproject.toml").read_text()
match = re.search(r'requires-python\s*=\s*"(.*?)"', text)
if not match:
    raise SystemExit("Could not find requires-python in pyproject.toml")
value = match.group(1)
version_match = re.search(r'(\d+\.\d+)', value)
if not version_match:
    raise SystemExit("Could not parse python version")
print(version_match.group(1))
"@ | python

$REQUIRED_PYTHON = $REQUIRED_PYTHON.Trim()
Write-Host "Required Python version: $REQUIRED_PYTHON"

# On Windows, Python is typically invoked as 'python' or via the launcher 'py -X.Y'
# Try the launcher first (most reliable on Windows), then fall back to 'python'
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PYTHON_BIN = "py"
    $PYTHON_ARGS = @("-$REQUIRED_PYTHON")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON_BIN = "python"
    $PYTHON_ARGS = @()
    $actualVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($actualVersion.Trim() -ne $REQUIRED_PYTHON) {
        Write-Error "ERROR: Python $REQUIRED_PYTHON required, but $actualVersion found on PATH"
        exit 1
    }
} else {
    Write-Error "ERROR: Python is not installed or not on PATH"
    exit 1
}

# --------------------------------------------
# Validate existing venv
# --------------------------------------------
$recreateVenv = $false

if (Test-Path ".venv") {
    Write-Host ""
    Write-Host "==> Existing .venv found"

    if (Test-Path ".venv\Scripts\python.exe") {
        $venvVersion = (& ".venv\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        Write-Host "Existing venv Python version: $venvVersion"

        if ($venvVersion -ne $REQUIRED_PYTHON) {
            Write-Host "Python version mismatch"
            $recreateVenv = $true
        }
    } else {
        Write-Host "Invalid venv"
        $recreateVenv = $true
    }
} else {
    $recreateVenv = $true
}

# --------------------------------------------
# Create/recreate venv
# --------------------------------------------
if ($recreateVenv) {
    Write-Host ""
    Write-Host "==> Creating virtual environment"

    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }

    & $PYTHON_BIN @PYTHON_ARGS -m venv .venv
}

# --------------------------------------------
# Activate venv
# --------------------------------------------
& .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------
# Install dependencies
# --------------------------------------------
Write-Host ""
Write-Host "==> Installing dependencies"

switch ($Mode) {
    "cpu" {
        pip install -e ".[dev]"
    }

    "nvidia" {
        pip install torch torchvision `
            --index-url https://download.pytorch.org/whl/cu130

        pip install -e ".[dev]"
    }

    "nvidia-legacy" {
        # CUDA 12.8 — for driver 570 (supports up to CUDA 12.8, not 13.0)
        pip install --force-reinstall torch torchvision `
            --index-url https://download.pytorch.org/whl/cu128

        pip install -e ".[dev]"

        # torch cu128 wheels pull in a newer numpy; pin it back to what the project requires
        pip install "numpy==2.2.6"
    }

    "amd-windows" {
        Write-Host ""
        Write-Host "==> Installing ROCm SDK (Step 1 of 3)"
        pip install --no-cache-dir `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

        Write-Host ""
        Write-Host "==> Installing PyTorch ROCm (Step 2 of 3)"
        pip install --no-cache-dir `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
            https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

        Write-Host ""
        Write-Host "==> Installing project (Step 3 of 3)"
        pip install -e ".[dev]"
    }
}

# --------------------------------------------
# Compile contracts
# --------------------------------------------
Write-Host ""
Write-Host "==> Compiling contracts"

python scripts/compile_contracts.py

Write-Host ""
Write-Host "============================================"
Write-Host "Setup complete"
Write-Host "============================================"
