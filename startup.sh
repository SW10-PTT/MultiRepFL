#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Startup script
# ============================================
#
# Usage:
#   ./startup.sh ganache    # auto_runner launches its own Ganache node
#   ./startup.sh anvil      # auto_runner launches its own Anvil node
#   ./startup.sh none       # no node; auto_runner uses RPC_URL from .env/.env.<ENV>
#
# The auto_runner now starts and manages its own blockchain node
# (see experiment/blockchain_launcher.py): it scans for a free port,
# sets RPC_URL, and tears the node down on exit. This script only
# activates the venv and launches the worker with the right flag.
#
# API_URL (and, for "none" mode, RPC_URL) must be set in the active
# env file (.env/.env.<ENV>, default .env/.env.ganache).

MODE="${1:-}"

case "$MODE" in
    ganache|anvil|none) ;;
    "")
        echo "Usage: ./startup.sh [ganache|anvil|none]"
        exit 1
        ;;
    *)
        echo "ERROR: unknown mode '$MODE' (expected ganache|anvil|none)"
        exit 1
        ;;
esac

# --------------------------------------------
# Move to repo root
# --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------
# Activate virtual environment
# --------------------------------------------
if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv does not exist"
    echo "Run setup.sh first"
    exit 1
fi

source .venv/bin/activate

# --------------------------------------------
# Compile contracts
# --------------------------------------------
echo "Compiling contracts..."
.venv/bin/python scripts/compile_contracts.py

# --------------------------------------------
# Run experiments
# --------------------------------------------
# auto_runner starts/stops its own node for ganache/anvil; "none" relies
# on an externally-provided RPC_URL from the env file.
echo "Starting experiments (mode: $MODE)..."

case "$MODE" in
    ganache) python experiment/auto_runner.py --ganache ;;
    anvil)   python experiment/auto_runner.py --anvil ;;
    none)    python experiment/auto_runner.py ;;
esac
