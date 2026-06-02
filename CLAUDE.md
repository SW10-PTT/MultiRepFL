# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenFL 2.0 is a federated learning research platform that integrates PyTorch-based distributed ML training with Ethereum smart contracts. It simulates Byzantine-resilient federated learning with on-chain reputation and incentive mechanisms.


## Purpose

Task-Conditioned Reputation for Participant Selection and
Robust Aggregation in Federated Learning Marketplaces
Many Federated Learning systems rely on reputation to determine participant selection
and participant influence on the FL process. Often, a single global reputation score is
used. However, in Federated Learning, different participants will excel at different types of
tasks. One participant may be excellent at vision tasks, but worse on NLP tasks, or on a
specific subtask of that category. Data quality is often task-specific, distribution-specific
and time-sensitive. This opens up a few issues. Participants who have built up a great
global reputation, will be favored for selection in all tasks, even though they may not be
able to provide value corresponding to their reputation for specific tasks, compared to
participants who have not been in the system for as long, but have great data for that
specific task type.
We propose a two-layer reputation system that separates Global integrity reputation and
Task-specific reputation to improve participant selection and thereby final model
performance compared to other selection methods. We will then experiment with our
proposed approach and compare it with the traditional approach of using global reputation
only.

## Rules

- Never run `git commit` or `git push`. Always leave committing and pushing to the user.
- When a commit or push is needed, tell the user what to commit and provide the exact command to run.
- A custom method called log exists for printing, taking a tag and the print message. Whenever a Print statement is needed, use log, along with an appropriate exsting or new tag. New tags can be found in print_config.py.

## Commands

### Setup
```bash
# CPU / NVIDIA (install GPU torch first if using NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130    # NVIDIA
pip install -e ".[dev]"
python3 scripts/compile_contracts.py   # Build ABI + bytecode from Solidity contracts

# AMD Linux:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
pip install -e ".[dev]"

# AMD Windows: see README section 3 for multi-step ROCm SDK + wheel install
```

### Running Experiments
```bash
ENV=ganache python ./experiment/experiment_runner.py
```

### Python Tests
```bash
pytest --cov=openfl tests/
```

### Solidity Tests (requires Foundry in WSL/Linux)
```bash
forge build
forge test
```

## Architecture

### Layers

**Experiment Layer** (`experiment/`)
- `experiment_configuration.py` — central config: participant counts, reward/collateral/punishment params, training hyperparams, contribution score strategy
- `experiments.py` — dataset-specific configs (CIFAR-10, MNIST)
- `experiment_runner.py` — orchestrates a full experiment end-to-end

**ML Layer** (`src/openfl/ml/pytorch_model.py`)
- `PytorchModel` — orchestrates federated learning simulation; manages participants, runs training rounds, evaluates contributions
- `Participant` — represents one FL participant; tracks collateral, reputation, attitude (good/bad/freerider/inactive), and submitted model hashes

**Contract Interaction Layer** (`src/openfl/contracts/`)
- `fl_manager.py` (`FLManager`) — deploys and manages `OpenFLManager`/`OpenFLModel` contracts; bridges Python ↔ blockchain
- `fl_challenge.py` (`FLChallenge`) — drives the FL round lifecycle: user registration, hashed weight submission, feedback exchange, reward/punishment dispatch; implements contribution scoring strategies

**Contribution Scoring Strategies** (selected via `ExperimentConfiguration`):
- `dotproduct` — matrix multiplication of weight vectors
- `naive` — accuracy-based
- `accuracy_loss` — combined accuracy + loss
- `accuracy_only` / `loss_only` / `loss_tolerance_aware` / `loss_tolerance_snap` — single-metric variants, and loss with tolerance
- 

**Blockchain / Web3 Layer** (`src/openfl/api/`, `contracts/`)
- `connection_helper.py` — RPC connection, ABI/bytecode loading, account init
- `OpenFLManager.sol` — deploys new FL model contracts per user
- `OpenFLModel.sol` — on-chain reputation system: registration, hashed weight submission, voting, punishments, rewards

### Environment Configuration

Environment files live in `.env/`. The active env is selected via `ENV=<identifier>` prefix (defaults to `ganache`).

Required variables:
- `RPC_URL` — blockchain RPC endpoint including port
- `PRIVATE_KEYS` — colon-separated private keys (only needed when `fork=false`, i.e., Sepolia; leave empty for Ganache fork mode)

### Ganache Setup

Ganache requires a workspace (not quickstart) with: gas limit set significantly above default, and high balance.

### Python Version

Python 3.12 is required (3.12.x). AMD GPU on Windows requires exactly Python 3.12 due to wheel availability.
