#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Project setup script
# ============================================

# Usage:
#   ./setup.sh [cpu|nvidia|amd-linux|amd-windows]
#
# Examples:
#   ./setup.sh cpu
#   ./setup.sh nvidia
#   ./setup.sh amd-linux

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  ./setup.sh [cpu|nvidia|amd-linux|amd-windows]"
    exit 1
fi

MODE="$1"

VALID_MODES=("cpu" "nvidia" "amd-linux" "amd-windows")

if [[ ! " ${VALID_MODES[*]} " =~ " ${MODE} " ]]; then
    echo "Invalid mode: $MODE"
    echo "Valid modes:"
    printf '  %s\n' "${VALID_MODES[@]}"
    exit 1
fi

echo "============================================"
echo "Setting up project ($MODE)"
echo "============================================"

# --------------------------------------------
# Ensure we're in repo root
# --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------
# Git cleanup
# --------------------------------------------
echo ""
echo "==> Resetting git state"

git checkout main
git fetch origin
git reset --hard origin/main

# --------------------------------------------
# Detect required Python version from pyproject.toml
# --------------------------------------------
echo ""
echo "==> Detecting required Python version"

if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: pyproject.toml not found"
    exit 1
fi

# Extract first version like 3.12 from requires-python
REQUIRED_PYTHON=$(
python3 - <<'PY'
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
PY
)

echo "Required Python version: $REQUIRED_PYTHON"

PYTHON_BIN="python${REQUIRED_PYTHON}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN is not installed"
    exit 1
fi

# --------------------------------------------
# Validate existing venv
# --------------------------------------------
RECREATE_VENV=false

if [[ -d ".venv" ]]; then
    echo ""
    echo "==> Existing .venv found"

    if [[ -f ".venv/bin/python" ]]; then
        VENV_PYTHON_VERSION=$(
            .venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        )

        echo "Existing venv Python version: $VENV_PYTHON_VERSION"

        if [[ "$VENV_PYTHON_VERSION" != "$REQUIRED_PYTHON" ]]; then
            echo "Python version mismatch"
            RECREATE_VENV=true
        fi
    else
        echo "Invalid venv"
        RECREATE_VENV=true
    fi
else
    RECREATE_VENV=true
fi

# --------------------------------------------
# Create/recreate venv
# --------------------------------------------
if [[ "$RECREATE_VENV" == true ]]; then
    echo ""
    echo "==> Creating virtual environment"

    rm -rf .venv

    "$PYTHON_BIN" -m venv .venv
fi

# --------------------------------------------
# Activate venv
# --------------------------------------------
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------
# Install dependencies
# --------------------------------------------
echo ""
echo "==> Installing dependencies"

case "$MODE" in
    cpu)
        pip install -e ".[dev]"
        ;;

    nvidia)
        pip install torch torchvision \
            --index-url https://download.pytorch.org/whl/cu130

        pip install -e ".[dev]"
        ;;

    amd-linux)
        pip install -e ".[dev]" \
            --extra-index-url https://download.pytorch.org/whl/rocm7.1
        ;;

    amd-windows)
        echo ""
        echo "AMD Windows setup must be run in PowerShell."
        echo ""
        echo "Run the following manually:"
        echo ""

        cat <<'EOF'
# Step 1 — ROCm SDK
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

# Step 2 — PyTorch
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

# Step 3 — Project
pip install -e ".[dev]"
EOF
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes:"
        echo "  cpu"
        echo "  nvidia"
        echo "  amd-linux"
        echo "  amd-windows"
        exit 1
        ;;
esac

# --------------------------------------------
# Compile contracts
# --------------------------------------------
echo ""
echo "==> Compiling contracts"

python3 scripts/compile_contracts.py

echo ""
echo "============================================"
echo "Setup complete"
echo "============================================"