# ============================================
# Startup script (Windows)
# ============================================
#
# NOTE: If you get "running scripts is disabled", run once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# Linux / macOS users: use startup.sh instead.

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
# Detect Ganache variant
# --------------------------------------------
$ganacheCmd = $null
$ganacheArgs = @()

if (Get-Command ganache -ErrorAction SilentlyContinue) {
    $ganacheCmd = "ganache"
    $ganacheArgs = @(
        "--wallet.totalAccounts", "30",
        "--wallet.defaultBalance", "1000000000",
        "--chain.chainId", "1337",
        "--server.host", "127.0.0.1",
        "--server.port", "8545"
    )
} elseif (Get-Command "ganache-cli" -ErrorAction SilentlyContinue) {
    $ganacheCmd = "ganache-cli"
    $ganacheArgs = @(
        "--accounts", "30",
        "--defaultBalanceEther", "1000000000",
        "--networkId", "1337",
        "--host", "127.0.0.1",
        "--port", "8545"
    )
} else {
    Write-Error "ERROR: Neither ganache nor ganache-cli is installed"
    exit 1
}

Write-Host "Using: $ganacheCmd"

# --------------------------------------------
# Prepare directories
# --------------------------------------------
New-Item -ItemType Directory -Force "env" | Out-Null

# --------------------------------------------
# Start Ganache
# --------------------------------------------
Write-Host "Starting Ganache..."

$ganacheProcess = Start-Process -FilePath $ganacheCmd `
    -ArgumentList $ganacheArgs `
    -RedirectStandardOutput "ganache.log" `
    -RedirectStandardError "ganache-err.log" `
    -PassThru -NoNewWindow

Write-Host "Ganache PID: $($ganacheProcess.Id)"

# --------------------------------------------
# Cleanup handler
# --------------------------------------------
$cleanupDone = $false
function Stop-Ganache {
    if (-not $script:cleanupDone) {
        $script:cleanupDone = $true
        Write-Host ""
        Write-Host "Stopping Ganache..."
        Stop-Process -Id $ganacheProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

# --------------------------------------------
# Wait for Ganache RPC
# --------------------------------------------
Write-Host "Waiting for Ganache RPC..."

$ganacheReady = $false

for ($i = 1; $i -le 30; $i++) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8545" `
            -Method Post `
            -ContentType "application/json" `
            -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' `
            -UseBasicParsing `
            -ErrorAction Stop | Out-Null
        $ganacheReady = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ganacheReady) {
    Stop-Ganache
    Write-Error "ERROR: Ganache failed to start"
    exit 1
}

Write-Host "Ganache is ready"

# --------------------------------------------
# Extract private keys (comma-separated, matching startup.sh)
# --------------------------------------------
$privateKeys = (Select-String -Path "ganache.log" -Pattern "0x[a-fA-F0-9]{64}" -AllMatches |
    ForEach-Object { $_.Matches } |
    ForEach-Object { $_.Value }) -join ","

# --------------------------------------------
# Create/update env/.env.ganache
# --------------------------------------------
$envFile = "env\.env.ganache"

if (-not (Test-Path $envFile)) {
    New-Item -ItemType File -Force $envFile | Out-Null
}

$content = Get-Content $envFile -Raw -ErrorAction SilentlyContinue
if (-not $content) { $content = "" }

if ($content -match '(?m)^PRIVATE_KEYS=') {
    $content = $content -replace '(?m)^PRIVATE_KEYS=.*', "PRIVATE_KEYS=`"$privateKeys`""
} else {
    $content += "`nPRIVATE_KEYS=`"$privateKeys`""
}

if ($content -notmatch '(?m)^RPC_URL=') {
    $content += "`nRPC_URL=`"http://127.0.0.1:8545`""
}

Set-Content -Path $envFile -Value $content.TrimStart() -NoNewline

Write-Host "Created env\.env.ganache"

# --------------------------------------------
# Run experiments
# --------------------------------------------
try {
    Write-Host "Starting experiments..."
    python experiment/auto_runner.py
} finally {
    Stop-Ganache
}
