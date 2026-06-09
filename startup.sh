#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Startup script
# ============================================
#
# Usage:
#   ./startup.sh ganache              # auto_runner launches its own Ganache node
#   ./startup.sh anvil                # auto_runner launches its own Anvil node
#   ./startup.sh none                 # no node; auto_runner uses RPC_URL from .env/.env.<ENV>
#
# Optional flag (any mode):
#   --threads=N   cap CPU threads per process (OMP/MKL/OpenBLAS) to N.
#                 Omit to keep the default (torch uses all cores). Handy when
#                 running multiple sessions on one machine to avoid CPU thrash.
#                 e.g. ./startup.sh ganache --threads=4
#
# The auto_runner now starts and manages its own blockchain node
# (see experiment/blockchain_launcher.py): it scans for a free port,
# sets RPC_URL, and tears the node down on exit. This script only
# activates the venv and launches the worker with the right flag.
#
# API_URL (and, for "none" mode, RPC_URL) must be set in the active
# env file (.env/.env.<ENV>, default .env/.env.ganache).

MODE=""
THREADS=""

# Parse args: one positional mode plus an optional --threads=N flag, in any order.
for arg in "$@"; do
    case "$arg" in
        ganache|anvil|none)
            MODE="$arg"
            ;;
        --threads=*)
            THREADS="${arg#*=}"
            ;;
        *)
            echo "ERROR: unknown argument '$arg'"
            echo "Usage: ./startup.sh [ganache|anvil|none] [--threads=N]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: ./startup.sh [ganache|anvil|none] [--threads=N]"
    exit 1
fi

# --threads is optional. When omitted, THREADS stays empty and no thread-count
# env vars are exported, so torch keeps its default (all cores) — identical to
# the previous behavior. When given, it must be a positive integer.
if [[ -n "$THREADS" ]] && ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --threads must be a positive integer (got '$THREADS')"
    exit 1
fi

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
# Optional CPU thread cap
# --------------------------------------------
# Limits how many CPU threads torch/OpenMP/MKL spawn per process. Useful when
# running several sessions on one box: without a cap each process grabs all
# cores, so N sessions oversubscribe the CPU and thrash. Omit --threads to keep
# the old behavior (no cap).
if [[ -n "$THREADS" ]]; then
    export OMP_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS"
    export OPENBLAS_NUM_THREADS="$THREADS"
    echo "CPU thread cap: $THREADS (OMP/MKL/OpenBLAS)"
fi

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
