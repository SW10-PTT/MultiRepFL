# OpenFL: Decentralized Federated Learning on Public Blockchain Systems

```
 ___ _   _ ____       ____  _____ _     
|_ _| | | |  _ \     |  _ \|  ___| |    
 | || |_| | |_) |____| | | | |_  | |    
 | ||  _  |  __/_____| |_| |  _| | |___ 
|___|_| |_|_|        |____/|_|   |_____|                      
```

# Getting started
## 1. Ganache
- Download Ganache
- Set up a workspace (Not quickstart)
- Set gas limit much higher than default, same with balance
- Set accounts to 8

## 2. Environment Variables
The project contains a .env file located in the .env folder, but supports easy replacement of this environmnent.
The project runs with the .env.ganache .env file by default. If another .env is preferred run the program with the 
``ENV=<env_file_identifier>`` prefix. Providing no ENV prefix and providing ``ENV=ganache`` is therefore equivalent.

In your Environment, you must have the following variables set:
```
RPC_URL="<RPC_URL from ganache or sepolia, including port>"
PRIVATE_KEYS="<Private keys from your accounts colon separated (for non-locally forked blockchain). If you have fork=true (using Ganache), there is no need to set private keys. Then just keep this variable empty>"
```

## 3. Requirements

> **Python 3.12 is required** for all platforms.

**CPU / NVIDIA:**
```bash
pip install -e ".[dev]"
```
For NVIDIA GPU acceleration, install the CUDA build of PyTorch first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"
```

**AMD GPU (Linux):**
```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/rocm7.1
```

**AMD GPU (Windows) — requires AMD driver 26.2.2:**

Step 1 — install the ROCm SDK (PowerShell):
```powershell
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
```

Step 2 — install PyTorch (PowerShell):
```powershell
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
```

Step 3 — install the project:
```bash
pip install -e ".[dev]"
```

Build the abi and bytecode files from the smart contracts:
```bash
python3 scripts/compile_contracts.py
```

## 4. Running an Experiment
The Experiment folder contains files for running experiments on different datasets.
To change the experiment setup, modify the experiment_configuration.py file.
To change the dataset, modify the experiments.py file.

The file experiments.py runs one such experiment and can be run with:
``ENV=ganache python ./experiment/experiment_runner.py``

The project can also be run from VS code, using one of the configuration profiles defined in the .vscode folder. E.g. ``Compile - Debug: Sample (1)``, to run the sample experiment.

## 5. Solidity testing
- Download Foundry in a Unix-like shell (WSL or Linux): 
  - `curl -L https://foundry.paradigm.xyz | bash && source ~/.bashrc`
  - `foundryup`
  - `forge soldeer install`
  
- To Test:
  - `forge build` 
  - `forge test`

# 6. Test Coverage
To get test coverage, run the following command: \
`pytest --cov=openfl tests/`

# 7. Data Partitioning

The dataset is split across users by a `DataPartition` strategy, selected via
`partition_strategy` on `ExperimentConfiguration`.

## Strategies

### `partition_strategy="global"` (default)

Stratified split across users using `data_percentages` (one share per user, must
sum to 100). Optional `label_rules` applies per-user `only_labels` filter and
`flip_map` (label-flipping for malicious users).

`allow_overlap=True` + `replication_factor>1.0` lets the same sample land under
multiple users; within-user dedup guarantees no user trains on the same image
twice. Disjoint mode (`allow_overlap=False`) is the default.

```python
ExperimentConfiguration(
    data_percentages=[30, 10, 15, 15, 10, 20],
    label_rules={
        0: {"only_labels": [0, 1, 2, 3, 4]},
        5: {"flip_map": {4: 9}},
    },
    allow_overlap=False,
)
```

### `partition_strategy="per_user"`

Each user gets a fully described `UserPartitionSpec`: total budget
(`data_percent`) plus optional per-class `label_distribution` (relative
weights), `only_labels` whitelist, and `flip_map`. Specs come from a single
JSON file (or in-memory dict).

Allocation is **total-budget-then-slice**: `data_percent` defines the user's
total share of the dataset; `label_distribution` slices that budget across
classes. Per-class supply check runs upfront — if total demand for any class
exceeds supply (or `supply * replication_factor` in overlap mode), partitioning
fails fast with a `ValueError`. Train/val split is stratified per class.

```python
ExperimentConfiguration(
    partition_strategy="per_user",
    per_user_partitions="experiment/partitions/example.json",
)
```

JSON format (see `experiment/partitions/example.json`):

```json
{
  "users": [
    {
      "id": 0,
      "data_percent": 12.0,
      "label_distribution": {"0": 0.5, "1": 0.5}
    },
    {
      "id": 1,
      "data_percent": 18.0,
      "only_labels": [5, 6, 7]
    },
    {
      "id": 2,
      "data_percent": 10.0,
      "only_labels": [4, 9],
      "flip_map": {"4": 9}
    }
  ]
}
```

Field reference per user:

| Field                | Required | Description                                                                  |
|----------------------|----------|------------------------------------------------------------------------------|
| `id` / `user_index`  | yes      | Zero-based data-user index. Must cover `0..number_of_data_users-1`.          |
| `data_percent`       | yes      | Total budget as % of dataset. Sum across users ≤ 100 (disjoint).             |
| `label_distribution` | no       | Per-class relative weights. Slices the total budget across listed classes.   |
| `only_labels`        | no       | Whitelist. Without `label_distribution`, budget is stratified across these.  |
| `flip_map`           | no       | Source→target label flip applied at read time (malicious user simulation).   |

Priority for class weighting: `label_distribution` > `only_labels` > full
stratified across all classes.

## Reproducibility

Both strategies are seeded by `seed` (master) and per-user `user_seeds`. The
active strategy + spec content is folded into `ExperimentConfiguration.get_finger_print`
and `User.finger_print`, so changing the partition triggers a fresh replay and
prevents cache hits against runs with different splits.

# 8. Setup and Startup Scripts

The repo includes scripts that handle setup and startup automatically.

## Setup

**Linux / WSL:**
```bash
./setup.sh [cpu|nvidia|nvidia-legacy|amd-linux|amd-windows]
```

**Windows (PowerShell):**
```powershell
.\windows-setup.ps1 [cpu|nvidia|nvidia-legacy|amd-windows]
```

| Mode | Description |
|------|-------------|
| `cpu` | No GPU |
| `nvidia` | NVIDIA GPU, driver 575+ (CUDA 13.0) |
| `nvidia-legacy` | NVIDIA GPU, driver 550-574 (CUDA 12.8) |
| `amd-linux` | AMD GPU on Linux |
| `amd-windows` | AMD GPU on Windows, requires AMD driver 26.2.2 |

The script creates a `.venv`, installs all dependencies, and compiles the smart contracts.

## Startup

**Linux / WSL:**
```bash
./startup.sh [ganache|anvil|none]
```

**Windows (PowerShell):**
```powershell
.\windows-startup.ps1 [ganache|anvil|none]
```

The startup script activates the virtual environment and launches the `auto_runner` worker. For `ganache`/`anvil`, `auto_runner` starts and manages its own blockchain node (scanning for a free port, setting `RPC_URL`, and shutting the node down on exit). Use `none` to skip launching a node and instead connect to an externally-provided `RPC_URL` from the active env file.

`API_URL` (and, for `none` mode, `RPC_URL`) must be set in the active env file (`.env/.env.<ENV>`, default `.env/.env.ganache`).

# 9. Generating Graphs

Experiments log results to `.pkl` files (multirep sessions ship a `session.pkl`
inside a `*.tar.gz`). Turn them into figures with:

```bash
# Aggregate comparison across experiments — averages every run of an experiment
# and compares the global-rep vs multi-rep variants. Scans
# experiment/data/FinishedRuns/ and writes PNGs to figures/aggregate/ (gitignored).
python analysis/multirep_aggregate_graphs.py

# Graphs for a single multirep session (tarball, session folder, or session.pkl):
python analysis/multirep_graphs.py <session.tar.gz | session-folder | session.pkl>

# Single-task experiment graphs (CSV/pkl pipeline):
python analysis/run_analysis.py
```

Each experiment directory under `FinishedRuns/` holds one or more run tarballs in
a `sessions/` subfolder; runs of the same experiment are averaged together, and
the `globalrep`/`multirep` variants of an identically-named experiment are paired
for comparison.

See [`analysis/README.md`](analysis/README.md) for the full graph catalogue,
expected folder layout, the data-split / free-rider / thesis graphs, and the
global-rep re-run caveat (a fixed replay bug means existing global-rep runs need
regenerating for clean numbers).