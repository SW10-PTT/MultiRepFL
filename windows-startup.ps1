# ============================================
# Startup script (Windows)
# ============================================
#
# Usage:
#   .\windows-startup.ps1 ganache
#   .\windows-startup.ps1 anvil
#
# NOTE: If you get "running scripts is disabled", run once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#
# Linux / macOS users: use startup.sh instead.

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("ganache", "anvil")]
    [string]$Mode
)

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
# Resolve node command and args
# --------------------------------------------
$nodeCmd = $null
$nodeArgs = @()

if ($Mode -eq "ganache") {
    if (Get-Command ganache -ErrorAction SilentlyContinue) {
        $nodeCmd = "ganache"
        $nodeArgs = @(
            "--wallet.totalAccounts", "30",
            "--wallet.defaultBalance", "1000000000",
            "--chain.chainId", "1337",
            "--server.host", "127.0.0.1",
            "--server.port", "8545"
        )
    } elseif (Get-Command "ganache-cli" -ErrorAction SilentlyContinue) {
        $nodeCmd = "ganache-cli"
        $nodeArgs = @(
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
} else {
    if (-not (Get-Command anvil -ErrorAction SilentlyContinue)) {
        Write-Error "ERROR: anvil is not installed (install Foundry)"
        exit 1
    }
    $nodeCmd = "anvil"
    $nodeArgs = @(
        "--accounts", "30",
        "--balance", "1000000000",
        "--chain-id", "1337",
        "--host", "127.0.0.1",
        "--port", "8545"
    )
}

Write-Host "Using: $nodeCmd"

# --------------------------------------------
# Prepare directories
# --------------------------------------------
New-Item -ItemType Directory -Force "env" | Out-Null

# --------------------------------------------
# Start node (hidden window to avoid corrupting this console)
# --------------------------------------------
Write-Host "Starting $nodeCmd..."

$cmdArgs = "/c $nodeCmd " + ($nodeArgs -join " ") + " > ganache.log 2>&1"
$nodeProcess = Start-Process -FilePath "cmd.exe" `
    -ArgumentList $cmdArgs `
    -PassThru `
    -WindowStyle Hidden

Write-Host "$nodeCmd PID: $($nodeProcess.Id)"

# --------------------------------------------
# Cleanup handler
# --------------------------------------------
$cleanupDone = $false
function Stop-Node {
    if (-not $script:cleanupDone) {
        $script:cleanupDone = $true
        Write-Host ""
        Write-Host "Stopping $($script:nodeCmd)..."
        taskkill /F /T /PID $script:nodeProcess.Id 2>$null
    }
}

# --------------------------------------------
# Wait for RPC
# --------------------------------------------
Write-Host "Waiting for RPC on :8545..."

$nodeReady = $false

for ($i = 1; $i -le 30; $i++) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8545" `
            -Method Post `
            -ContentType "application/json" `
            -Body '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' `
            -UseBasicParsing `
            -ErrorAction Stop | Out-Null
        $nodeReady = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $nodeReady) {
    Stop-Node
    Write-Error "ERROR: $nodeCmd failed to start"
    exit 1
}

Write-Host "$nodeCmd is ready"

# --------------------------------------------
# Extract private keys
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
    Stop-Node
}
