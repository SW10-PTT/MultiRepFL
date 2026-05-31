#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Startup script
# ============================================

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

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

# --------------------------------------------
# Detect Ganache variant
# --------------------------------------------
if command -v ganache >/dev/null 2>&1; then
    GANACHE_CMD="ganache"

    GANACHE_ARGS=(
        --wallet.totalAccounts 30
        --wallet.defaultBalance 1000000000
        --chain.chainId 1337
        --server.host 127.0.0.1
        --server.port 8545
    )

elif command -v ganache-cli >/dev/null 2>&1; then
    GANACHE_CMD="ganache-cli"

    GANACHE_ARGS=(
        --accounts 30
        --defaultBalanceEther 1000000000
        --networkId 1337
        --host 127.0.0.1
        --port 8545
    )

else
    echo "ERROR: Neither ganache nor ganache-cli is installed"
    exit 1
fi

echo "Using: $GANACHE_CMD"

# --------------------------------------------
# Prepare directories
# --------------------------------------------
mkdir -p .env

# --------------------------------------------
# Start Ganache
# --------------------------------------------
echo "Starting Ganache..."

"$GANACHE_CMD" "${GANACHE_ARGS[@]}" \
    > ganache.log 2>&1 &

GANACHE_PID=$!

echo "Ganache PID: $GANACHE_PID"

# --------------------------------------------
# Cleanup handler
# --------------------------------------------
cleanup() {
    echo ""
    echo "Stopping Ganache..."

    kill "$GANACHE_PID" >/dev/null 2>&1 || true
}

trap cleanup EXIT

# --------------------------------------------
# Wait for Ganache RPC
# --------------------------------------------
echo "Waiting for Ganache RPC..."

GANACHE_READY=false

for i in {1..30}; do
    if curl -s \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' \
        http://127.0.0.1:8545 \
        >/dev/null 2>&1; then

        GANACHE_READY=true
        break
    fi

    sleep 1
done

if [[ "$GANACHE_READY" != true ]]; then
    echo "ERROR: Ganache failed to start"
    exit 1
fi

echo "Ganache is ready"

# --------------------------------------------
# Extract private keys
# --------------------------------------------
PRIVATE_KEYS=$(
    grep -oE '0x[a-fA-F0-9]{64}' ganache.log \
    | paste -sd "," -
)

# --------------------------------------------
# Create/update .env/.env.ganache (preserves existing keys like API_URL)
# --------------------------------------------
ENV_FILE=".env/.env.ganache"

python - "$PRIVATE_KEYS" "$ENV_FILE" <<'PYEOF'
import sys
from pathlib import Path

private_keys = sys.argv[1]
env_file = Path(sys.argv[2])
content = env_file.read_text(encoding="utf-8") if env_file.exists() else ""

lines = [l for l in content.splitlines() if l.strip()]
new_lines = []
found_pk = found_rpc = False

for line in lines:
    if line.startswith("PRIVATE_KEYS="):
        new_lines.append(f'PRIVATE_KEYS="{private_keys}"')
        found_pk = True
    elif line.startswith("RPC_URL="):
        new_lines.append('RPC_URL="http://127.0.0.1:8545"')
        found_rpc = True
    else:
        new_lines.append(line)

if not found_pk:
    new_lines.append(f'PRIVATE_KEYS="{private_keys}"')
if not found_rpc:
    new_lines.append('RPC_URL="http://127.0.0.1:8545"')

env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
PYEOF

echo "Created .env/.env.ganache"

# --------------------------------------------
# Run experiments
# --------------------------------------------
echo "Starting experiments..."

python experiment/auto_runner.py