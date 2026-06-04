# ============================================
# Startup script (Windows)
# ============================================
#
# Usage:
#   .\windows-startup.ps1 ganache   # auto_runner launches its own Ganache node
#   .\windows-startup.ps1 anvil     # auto_runner launches its own Anvil node
#   .\windows-startup.ps1 none      # no node; auto_runner uses RPC_URL from .env/.env.<ENV>
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
    [string]$Mode = ""
)

if (-not $Mode) {
    Write-Host "Usage: .\windows-startup.ps1 [ganache|anvil|none]"
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
