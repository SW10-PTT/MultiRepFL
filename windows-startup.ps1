# ============================================
# Startup script (Windows)
# ============================================
#
# Usage:
#   .\windows-startup.ps1 ganache   # auto_runner launches its own Ganache node
#   .\windows-startup.ps1 anvil     # auto_runner launches its own Anvil node
#   .\windows-startup.ps1 none      # no node; auto_runner uses RPC_URL from .env/.env.<ENV>
#
# Optional flag (any mode):
#   -Threads N   cap CPU threads per process (OMP/MKL/OpenBLAS) to N.
#                Omit to keep the default (torch uses all cores). Handy when
#                running multiple sessions on one machine to avoid CPU thrash.
#                e.g. .\windows-startup.ps1 ganache -Threads 4
#
# The auto_runner now starts and manages its own blockchain node
# (see experiment/blockchain_launcher.py): it scans for a free port,
# sets RPC_URL, and tears the node down on exit. This script only
# activates the venv and launches the worker with the right flag.
#
# API_URL (and, for "none" mode, RPC_URL) must be set in the active
# env file (.env\.env.<ENV>, default .env\.env.ganache).
#
# NOTE: If you get "running scripts is disabled", run once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# Linux / macOS users: use startup.sh instead.

param(
    [ValidateSet("ganache", "anvil", "none")]
    [string]$Mode = "",

    # Optional: cap CPU threads per process (OMP/MKL/OpenBLAS). 0 = no cap (default).
    [int]$Threads = 0
)

if (-not $Mode) {
    Write-Host "Usage: .\windows-startup.ps1 [ganache|anvil|none] [-Threads N]"
    exit 1
}

if ($Threads -lt 0) {
    Write-Error "ERROR: -Threads must be a positive integer (got '$Threads')"
    exit 1
}

$ErrorActionPreference = "Stop"

# --------------------------------------------
# Move to repo root
# --------------------------------------------
Set-Location $PSScriptRoot

# --------------------------------------------
# Activate virtual environment
# --------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Error "ERROR: .venv does not exist. Run windows-setup.ps1 first."
    exit 1
}

& .venv\Scripts\Activate.ps1

# --------------------------------------------
# Optional CPU thread cap
# --------------------------------------------
# Limits how many CPU threads torch/OpenMP/MKL spawn per process. Useful when
# running several sessions on one box: without a cap each process grabs all
# cores, so N sessions oversubscribe the CPU and thrash. Omit -Threads (or pass
# 0) to keep the old behavior (no cap).
if ($Threads -gt 0) {
    $env:OMP_NUM_THREADS = "$Threads"
    $env:MKL_NUM_THREADS = "$Threads"
    $env:OPENBLAS_NUM_THREADS = "$Threads"
    Write-Host "CPU thread cap: $Threads (OMP/MKL/OpenBLAS)"
}

# --------------------------------------------
# Compile contracts
# --------------------------------------------
Write-Host "Compiling contracts..."
& .venv\Scripts\python.exe scripts\compile_contracts.py

# --------------------------------------------
# Run experiments
# --------------------------------------------
# auto_runner starts/stops its own node for ganache/anvil; "none" relies
# on an externally-provided RPC_URL from the env file.
Write-Host "Starting experiments (mode: $Mode)..."

switch ($Mode) {
    "ganache" { python experiment/auto_runner.py --ganache }
    "anvil"   { python experiment/auto_runner.py --anvil }
    "none"    { python experiment/auto_runner.py }
}
