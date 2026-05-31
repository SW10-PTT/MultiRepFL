import json
import os
import random
import re
import sys
import tarfile
import traceback
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment.experiment_configuration import ExperimentConfiguration

from datetime import datetime
from typing import List
import requests
import experiment.experiment_runner as ExperimentRunner
from experiment.experiment_runner import setup_connection
from experiment.multirep.MultirepLogger import (
    MultirepLogger, pack_session_tarball, copy_remote_task_files, load_task_pkl_tables,
)
from experiment.multirep.MultirepPreset import MultirepPreset
from experiment.multirep.MultirepRunConfig import MultirepRunConfig
from experiment.multirep.training_mode import TrainingMode
from openfl.contracts import FLManager as Manager
from web3 import Web3
from openfl.utils.types.User import User
from openfl.utils.printer import log, set_log_file, set_enabled_tags
from openfl.utils.W3Helper import get_PRIVKEYS, get_RPC_Endpoint
from openfl.api import globals as fl_globals
from openfl.api.globals import ReplayMode
from analysis.ExperimentLogger import ExperimentLogger
from openfl.utils.async_writer import AsyncWriter


# ---------------------------------------------------------------------------
# Scoring configuration — mirrors the smart contract's getTopN selection logic.
# Used to predict participant selection for fingerprinting / RunRepo caching.
# ---------------------------------------------------------------------------

_WAD = 10 ** 18  # all on-chain rep values are WAD-scaled

# EWMA constants — mirror JobListing.sol exactly.
_ALPHA = int(2e17)                    # forgetting factor for running mean + variance
_N_BLEND = int(2e17)                  # smoothing on final ContribScore → TaskRep
_N_0 = 5                              # maturity offset
_LAMBDA = 20                          # variance penalty weight
_GAIN_CAP_MULTIPLIER = 2
_STAKE_WAD = int(1e18)                # collateral (hardcoded to 1 ETH, matching Solidity)
_INTEGRITY_LEARNING_RATE = int(2e17)  # GIR EWMA learning rate


# ---------------------------------------------------------------------------
# Preset file — fill in before running
# ---------------------------------------------------------------------------

preset_file = "experiment/presets/task-hopper-showcase-remote.json"

# ---------------------------------------------------------------------------
# Output directory for multirep sessions
# ---------------------------------------------------------------------------

MULTIREP_DATA_DIR = Path(__file__).resolve().parent / "data" / "multirepData"

# CSV writer config (mirrors auto_runner.py)
_OUTPUTHEADERS = [
    "round", "time", "globalAcc", "globalLoss", "GRS",
    "accAvgPerUser", "lossAvgPerUser", "rewards",
    "conctractBalanceRewards", "punishments", "contributionScores",
    "feedbackMatrix", "disqualifiedUsers", "userStatuses",
    "GasTransactions", "Contrib",
]
_WRITERBUFFERSIZE = 200


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_user_score(user: User, task_type: int, q_weight: float = 0.0, tr_weight: int = 6, gir_weight: int = 4) -> int:
    # score = (taskRep * tr_weight + gir * gir_weight) / (tr_weight + gir_weight) + q_weight * q
    # All values are WAD-scaled; no conversion here.
    denom = tr_weight + gir_weight
    base = user.task_rep.get(task_type, 0) * tr_weight + user.global_integrity_rep * gir_weight
    normal_weight = base // denom
    q = user.q_value.get(task_type, 0)
    return int(normal_weight + q_weight * q)


def getTopN(users: List[User], n: int, task_type: int, q_weight: float = 0.0, tr_weight: int = 6, gir_weight: int = 4) -> List[User]:
    """Mirror the smart contract's participant selection for fingerprinting."""
    fps = {u: u.finger_print for u in users}
    scores = [(compute_user_score(u, task_type, q_weight, tr_weight, gir_weight), u) for u in users]
    scores.sort(key=lambda x: (-x[0], fps[x[1]]))
    selected = [u for _, u in scores[:n]]
    selected_set = {u.address for u in selected}

    # Register every user's finger_print → label so the replay diff can resolve names.
    for _, u in scores:
        label = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else f"User {u.number}"
        fl_globals.fp_user_labels[fps[u]] = label

    log("multirep", f"Selection (top {n} of {len(users)}, task_type={task_type}, q_weight={q_weight / _WAD:.4f}, tr={tr_weight}, gir={gir_weight}):")
    log("multirep", f"  {'Name':<16} {'TR':>8} {'GIR':>8} {'Q':>8} {'Score':>10}  fp[:8]  sel")
    for score, u in scores:
        marker = "YES" if u.address in selected_set else "no"
        name = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else f"User {u.number}"
        tr = u.task_rep.get(task_type, 0) / _WAD
        gir = u.global_integrity_rep / _WAD
        q = u.q_value.get(task_type, 0) / _WAD
        fp8 = fps[u][:8]
        log("multirep", f"  {name:<16} {tr:>8.4f} {gir:>8.4f} {q:>8.4f} {score / _WAD:>10.4f}  {fp8}  {marker}")

    return selected


# ---------------------------------------------------------------------------
# On-chain reputation helpers
# ---------------------------------------------------------------------------

def get_task_type(dataset: str) -> int:
    """Map dataset name to TaskType enum value from Types.sol."""
    from openfl.utils.types.TrainingSpecsJobListing import TaskType
    return int(TaskType.from_dataset_name(dataset))


def update_users_from_reps(users: List[User], reps, task_type: int) -> None:
    users_by_address = {u.address.lower(): u for u in users}
    users_by_guid = {u.guid: u for u in users if u.guid is not None}
    for rep in reps:
        if isinstance(rep, dict):
            address = rep.get("address", "")
            task_rep = rep["taskRep"]
            global_integrity_rep = rep["globalIntegrityRep"]
            total_contrib_score = rep["totalContribScore"]
            q_value = rep["qValue"]
            balance = rep.get("balance", 0)
            guid = rep.get("guid")
        else:
            address = rep[0]
            task_rep, global_integrity_rep, total_contrib_score, q_value = rep[1], rep[2], rep[3], rep[4]
            balance = rep[5] if len(rep) > 5 else 0
            guid = None
        user = (users_by_guid.get(guid) if guid else None) or users_by_address.get(address.lower())
        if user is None:
            continue
        user.task_rep[task_type] = task_rep
        user.global_integrity_rep = global_integrity_rep
        user.total_contrib_score = total_contrib_score
        user.q_value[task_type] = q_value
        user.balance = balance


# ---------------------------------------------------------------------------
# Q-value update (patience / selection pressure formula)
# ---------------------------------------------------------------------------

def _compute_q_updates(users: List[User], selected_users: List[User], task_type: int) -> dict:
    """Return {address: new_q} using pre-run Q values — does not modify users.

    Formula (k = selected, n = total):
      selected:     q_new = max(0, q_old + k/n - 1)
      not selected: q_new = max(0, q_old + k/n)
    """
    k = len(selected_users)
    n = len(users)
    ratio = k / n if n > 0 else 0.0
    selected_addrs = {u.address for u in selected_users}
    selected_guids = {u.guid for u in selected_users if u.guid is not None}
    updates = {}
    for user in users:
        q_old = user.q_value.get(task_type, 0) / _WAD
        is_selected = user.address in selected_addrs or (
            user.guid is not None and user.guid in selected_guids
        )
        delta = ratio - (1.0 if is_selected else 0.0)
        updates[user.address] = int(max(0.0, q_old + delta) * _WAD)
    return updates


def _apply_q_updates(users: List[User], q_updates: dict, task_type: int) -> None:
    for user in users:
        if user.address in q_updates:
            user.q_value[task_type] = q_updates[user.address]


# ---------------------------------------------------------------------------
# EWMA helpers — mirror JobListing.sol _transform_delta / _updateRunningStats /
# _computeConfidence / _updateContribScore / _updateIntegrityRep exactly
# (integer arithmetic, WAD-scaled).
# ---------------------------------------------------------------------------

def _transform_delta(delta: int, stake: int, reward: int, nr_active: int) -> int:
    max_gain = (_GAIN_CAP_MULTIPLIER * reward) // nr_active if nr_active > 0 else 0
    range_ = stake + max_gain
    if range_ == 0:
        return 0
    shifted = delta + stake
    if shifted <= 0:
        return 0
    if shifted >= range_:
        return _WAD
    return (shifted * _WAD) // range_


def _update_running_stats(contrib_score: int, prior_mean: int, prior_m2: int, k: int):
    new_mean = contrib_score if k <= 1 else ((_WAD - _ALPHA) * prior_mean + _ALPHA * contrib_score) // _WAD
    abs_delta = abs(contrib_score - prior_mean)
    abs_delta2 = abs(contrib_score - new_mean)
    new_m2 = ((_WAD - _ALPHA) * prior_m2) // _WAD + (_ALPHA * abs_delta * abs_delta2) // (_WAD * _WAD)
    return new_mean, new_m2


def _compute_confidence(k: int, s_k: int) -> int:
    if k == 0:
        return 0
    maturity = (k * _WAD) // (k + _N_0)
    stability = (_WAD * _WAD) // (_WAD + _LAMBDA * s_k)
    return (maturity * stability) // _WAD


def _update_contrib_score(prior_task_rep: int, confidence: int, contrib_score: int) -> int:
    weighted = (confidence * contrib_score) // _WAD
    return ((_WAD - _N_BLEND) * prior_task_rep + _N_BLEND * weighted) // _WAD


def _update_integrity_rep(prior_gir: int, pos_votes: int, total_votes: int) -> int:
    """Mirror JobListing._updateIntegrityRep: GIR = EWMA of (posVotes/totalVotes)²."""
    if total_votes == 0:
        v = 0
    else:
        ratio = (pos_votes * _WAD) // total_votes
        v = (ratio * ratio) // _WAD
    return ((_WAD - _INTEGRITY_LEARNING_RATE) * prior_gir + _INTEGRITY_LEARNING_RATE * v) // _WAD


# ---------------------------------------------------------------------------
# TRS (replay rep update)
# ---------------------------------------------------------------------------

def _apply_trs_reps(users: List[User], trs: list, task_type: int, manager, reward: int) -> None:
    """Apply EWMA TaskRep update on-chain and in Python for replayed runs.

    trs format: (guid, delta_task_rep, delta_balance, pos_votes, total_votes).
    Mirrors the Solidity JobListing._applyContribAndStats EWMA chain so TaskRep
    stays in [0, WAD] regardless of the number of tasks completed.
    """
    nr_active = len(trs)
    users_by_guid = {u.guid: u for u in users if u.guid is not None}
    log("multirep", f"{'User':<16} {'delta':>12} {'ContribScore':>14} {'confidence':>12} {'TaskRep→':>10} {'GIR→':>10} {'Balance(ETH)':>14}")
    for entry in trs:
        guid, delta, delta_balance = str(entry[0]), entry[1], entry[2]
        pos_votes   = entry[3] if len(entry) > 3 else 0
        total_votes = entry[4] if len(entry) > 4 else 0
        user = users_by_guid.get(guid)
        if user is None:
            continue

        prior_task_rep = user.task_rep.get(task_type, 0)
        prior_mean, prior_m2 = manager.get_task_rep_calc_state(user.address, task_type)
        k = user.task_count.get(task_type, 0) + 1

        contrib_score = _transform_delta(delta, _STAKE_WAD, reward, nr_active)
        new_mean, new_m2 = _update_running_stats(contrib_score, prior_mean, prior_m2, k)
        confidence = _compute_confidence(k, new_m2)
        new_task_rep = _update_contrib_score(prior_task_rep, confidence, contrib_score)

        # GIR: seed at WAD for first-task users (mirrors Solidity effectivePrior logic).
        prior_gir = user.global_integrity_rep or _WAD
        new_gir = _update_integrity_rep(prior_gir, pos_votes, total_votes)
        new_balance = max(0, user.balance + delta_balance)

        name = user.partition_spec.name if (user.partition_spec and user.partition_spec.name) else f"User {user.number}"
        log("multirep", f"  {name:<14} {delta / _WAD:>12.4f} {contrib_score / _WAD:>14.4f} {confidence / _WAD:>12.4f} "
                        f"{prior_task_rep / _WAD:>6.4f}→{new_task_rep / _WAD:.4f} "
                        f"{prior_gir / _WAD:>6.4f}→{new_gir / _WAD:.4f} "
                        f"{user.balance / _WAD:>6.4f}→{new_balance / _WAD:.4f}")

        manager.set_user_task_rep(user.address, task_type, new_task_rep)
        manager.set_task_rep_calc_state(user.address, task_type, new_mean, new_m2)
        manager.set_user_integrity_rep(user.address, new_gir)
        manager.set_user_balance(user.address, new_balance)

        # task_count is not persisted on-chain; keep it Python-side only.
        user.task_count[task_type] = k


# ---------------------------------------------------------------------------
# Partition filtering
# ---------------------------------------------------------------------------

def log_user_reputations(users: List[User], task_type: int, selected_users: List[User], q_weight: float = 0.0, tr_weight: int = 6, gir_weight: int = 4) -> None:
    """Log reputation fields and selection status for every user."""
    selected_set = {u.address for u in selected_users}
    log("multirep", "─" * 96)
    log("multirep", f"{'User':<12} {'Address':<20}  {'TaskRep':>8} {'GIR':>8} {'Balance(ETH)':>13} {'Q':>7} {'Score':>8}  {'Selected':>8}")
    log("multirep", "─" * 96)
    for u in users:
        score = compute_user_score(u, task_type, q_weight, tr_weight, gir_weight)
        tr = u.task_rep.get(task_type, 0) / _WAD
        gir = u.global_integrity_rep / _WAD
        balance = u.balance / _WAD
        q = u.q_value.get(task_type, 0) / _WAD
        score_display = score / _WAD
        selected = "YES" if u.address in selected_set else "no"
        name = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else None
        label = name if name else f"User {u.number}"
        addr = u.address[:20] if u.address else "N/A"
        log("multirep", f"{label:<12} {addr:<20}  {tr:>8.4f} {gir:>8.4f} {u.balance / _WAD:>13.4f} {q:>7.3f} {score_display:>8.4f}  {selected:>8}")
    log("multirep", "─" * 96)


def _sync_balances_from_challenge(users: List[User], experiment, manager) -> None:
    """Local-run path: read each participant's globalReputationScore from the
    settled challenge contract and write it into the manager's Balance slot."""
    challenge_contract = experiment.model.contract
    users_by_address = {u.address.lower(): u for u in users}
    for _, user in users_by_address.items():
        try:
            challenge_user = challenge_contract.functions.users(
                Web3.to_checksum_address(user.address)
            ).call()
            # OpenFLChallenge.User tuple field order: weightedContribScore[0], globalReputationScore[1].
            grs = challenge_user[1]
            manager.set_user_balance(user.address, grs)
        except Exception as e:
            log("multirep", f"[warn] balance sync failed for {user.address[:10]}...: {e}")


def filter_partitions_for_users(selected_users: List[User]) -> dict:
    """Return a flat {user_index: UserPartitionSpec} dict for the selected users.

    The ANY_DATASET wrapping required by ExperimentConfiguration is added by
    to_experiment_config_with_partitions, not here, so callers receive a clean
    map without internal implementation details leaking through.
    """
    return {
        user.partition_spec.user_index: user.partition_spec
        for user in selected_users
        if user.partition_spec is not None
    }


# ---------------------------------------------------------------------------
# RunRepo cache lookup
# ---------------------------------------------------------------------------

def _apply_cached_reps(users: List[User], cached_run: dict, task_type: int) -> None:
    """Apply reputation data from a cached API run response to user objects."""
    reps_data = cached_run.get("reputations", [])
    if not reps_data:
        log("multirep", "[warn] No reputation data in cached run — rep state unchanged.")
        return
    update_users_from_reps(users, reps_data, task_type)


def _fetch_cached_run(fingerprint: str):
    """Return the first API run for this fingerprint, or None.
    Used by LOCAL and REMOTE modes for the quick early-exit cache check."""
    runs = _fetch_runs_by_fingerprint(fingerprint)
    return runs[0] if runs else None


def _fetch_runs_by_fingerprint(fingerprint: str) -> list:
    """Return ALL completed run dicts for this fingerprint from the API.
    Handles both list and single-object API responses."""
    api_url = os.environ.get("API_URL")
    if not api_url:
        return []
    try:
        res = requests.get(f"{api_url}/runs/by-fingerprint/{fingerprint}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                return data
            if data:
                return [data]
    except Exception as e:
        log("multirep", f"[warn] fingerprint run lookup failed: {e}")
    return []


def _register_run(api_url: str, fingerprint: str, config: str) -> str | None:
    try:
        res = requests.post(f"{api_url}/runs/local", json={"fingerprint": fingerprint, "config": config}, timeout=10)
        if res.status_code == 200:
            return res.json().get("id")
        log("multirep", f"[warn] run registration returned {res.status_code}: {res.text[:120]}")
    except Exception as e:
        log("multirep", f"[warn] run registration failed: {e}")
    return None


def _upload_run(fingerprint: str, filename: Path, config: str) -> None:
    """Register a new run in the API, create a tarball of filename, and upload it."""
    api_url = os.environ.get("API_URL")
    if not api_url:
        return

    run_id = _register_run(api_url, fingerprint, config)
    if run_id is None:
        log("multirep", "[warn] Could not register run — skipping upload.")
        return

    try:
        url_res = requests.post(f"{api_url}/runs/{run_id}/upload-url", json=str(filename), timeout=10)
        url_res.raise_for_status()
        upload_url = url_res.json()["uploadUrl"]

        archive_path = filename.with_suffix(".tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(filename, filename.name)

        with open(archive_path, "rb") as f:
            put_res = requests.put(
                upload_url, data=f,
                headers={"Content-Type": "application/gzip"},
                timeout=300,
            )
        put_res.raise_for_status()

        requests.post(f"{api_url}/runs/{run_id}/complete", json={"RunId": str(run_id)})
        log("multirep", f"Run uploaded successfully (run_id={run_id}).")
    except Exception as e:
        log("multirep", f"[warn] Upload failed: {e}")


# ---------------------------------------------------------------------------
# Training-mode dispatch
# ---------------------------------------------------------------------------
fallback_count = 0

def _rep_internals_after_task(users: List[User], manager, task_type: int) -> tuple[dict, dict, dict, dict]:
    """Return (confidence, k, running_c_mean, m2) dicts keyed by user address.

    Reads getTaskRepCalcState and getUserRep from the chain for each user.
    Falls back to Python-side task_count and zero values on error.
    """
    confidence, k_map, mean_map, m2_map = {}, {}, {}, {}
    for u in users:
        try:
            c_mean, m2 = manager.get_task_rep_calc_state(u.address, task_type)
            _, _, nr_tasks = manager.contract.functions.getUserRep(u.address, task_type).call()
            conf = _compute_confidence(nr_tasks, m2)
            confidence[u.address] = conf / _WAD
            k_map[u.address] = nr_tasks
            mean_map[u.address] = c_mean / _WAD
            m2_map[u.address] = m2 / _WAD
            if nr_tasks > 0:
                u.task_count[task_type] = nr_tasks
        except Exception:
            confidence[u.address] = 0.0
            k_map[u.address] = u.task_count.get(task_type, 0)
            mean_map[u.address] = 0.0
            m2_map[u.address] = 0.0
    return confidence, k_map, mean_map, m2_map


def _run_preset(preset: MultirepRunConfig, exp_config, all_users, manager, fingerprint, experiment_name,
                writer=None, logger=None, path=None):
    from experiment.multirep.training_mode import TrainingMode

    if preset.training_mode == TrainingMode.REMOTE:
        return _run_remote(preset, exp_config, all_users, manager, fingerprint, experiment_name,
                           writer=writer, logger=logger, path=path)

    # LOCAL
    fl_globals.reuse_runs = ReplayMode.Record
    return ExperimentRunner.run_experiment(
        preset.dataset,
        exp_config,
        writer=writer,
        logger=logger,
        path=path,
        prebuilt_users=all_users,
        prebuilt_manager=manager,
    )


def _run_remote(preset: MultirepRunConfig, exp_config: ExperimentConfiguration, all_users, manager, fingerprint: str, experiment_name,
                writer=None, logger=None, path=None):
    """Submit to the remote API (or reuse a pooled run), replay locally.
    Falls back to LOCAL if anything goes wrong.

    remote_pool_size controls reuse probability:
      0  → always submit a new run
      N  → build a list of length N, fill from existing runs, pick a random
           slot; if the slot is non-None reuse that run, else submit new.
    """
    from experiment.multirep.remote_client import (
        run_remote_and_setup_replay, fetch_run_download_url,
        download_tarball, extract_and_register_runrepo,
    )

    try:
        pool_size = preset.remote_pool_size
        use_existing_run = None

        if pool_size > 0:
            pool = [None] * pool_size
            runs = _fetch_runs_by_fingerprint(fingerprint)
            for i, run in enumerate(runs[:pool_size]):
                pool[i] = run
            idx = random.randint(0, pool_size - 1)
            use_existing_run = pool[idx]
            log("multirep", f"[REMOTE] pool_size={pool_size}, existing={len(runs)}, idx={idx}, reuse={use_existing_run is not None}")

        if use_existing_run is not None:
            run_id = str(use_existing_run.get("id", "unknown"))
            download_url = fetch_run_download_url(run_id)
            dest = Path(__file__).resolve().parent / "data" / "remote_runs" / run_id
            archive = download_tarball(download_url, dest)
            extract_and_register_runrepo(archive, dest)
        else:
            run_remote_and_setup_replay(
                exp_config,
                fingerprint=fingerprint,
                name=experiment_name or f"multirep-{preset.dataset}",
            )

        return ExperimentRunner.run_experiment(
            preset.dataset,
            exp_config,
            writer=writer,
            logger=logger,
            path=path,
            prebuilt_users=all_users,
            prebuilt_manager=manager,
        )

    except Exception as e:
        log("multirep", f"[REMOTE] Failed — falling back to LOCAL.\nReason: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        fl_globals.reuse_runs = ReplayMode.Record
        # Restore repo_dir to task_dir so the JSON lands there, not in the
        # remote extraction dir that extract_and_register_runrepo may have set.
        if path is not None:
            fl_globals.repo_dir = str(path.parent)
        return ExperimentRunner.run_experiment(
            preset.dataset,
            exp_config,
            writer=writer,
            logger=logger,
            path=path,
            prebuilt_users=all_users,
            prebuilt_manager=manager,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global fallback_count
    preset = MultirepPreset.from_file(preset_file)
    tasks = preset.tasks
    partition_file = preset.partition_file
    training_mode = preset.training_mode

    if not tasks:
        log("multirep", "No tasks configured — nothing to run.")
        return

    # ---------------------------------------------------------------------- #
    # Session setup: log file + output folder + MultirepLogger                #
    # ---------------------------------------------------------------------- #
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_file(str(log_dir / f"multirep_{session_ts}.log"))

    preset_name_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", preset.name)
    session_dir = MULTIREP_DATA_DIR / f"{preset_name_safe}_{session_ts}"
    tasks_dir = session_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    with open(preset_file, "r", encoding="utf-8") as _f:
        preset_dict = json.load(_f)

    multirep_logger = MultirepLogger(
        session_id=str(uuid.uuid4()),
        preset_name=preset.name,
        session_timestamp=session_ts,
        preset_dict=preset_dict,
    )
    log("multirep", f"Session folder: {session_dir}")

    # ---------------------------------------------------------------------- #
    # Build users and deploy shared manager                                   #
    # ---------------------------------------------------------------------- #
    first_task = tasks[0]

    # Load ALL partition specs from the JSON once.  Users are created from
    # this full pool and keep their data partitions for the entire session.
    full_config = first_task.to_experiment_config(partition_file)
    full_config.name = preset.name

    # Enable print tags immediately so log() calls are visible on the terminal
    # throughout the full session (setup, remote polling, replay, etc.) rather
    # than only after the first run_experiment call activates them.
    set_enabled_tags(full_config.enabled_prints)
    log("multirep", f"=== MultiRep session started {session_ts} — {preset.name} ===")
    all_users = ExperimentRunner.build_users(full_config)

    # Deploy the manager contract once before the loop so it persists across
    # all tasks, including those that are skipped via the RunRepo cache.
    setup_connection(full_config)
    publisher = all_users[0]
    rpc = get_RPC_Endpoint()
    privkeys = get_PRIVKEYS(full_config)
    manager = Manager(publisher, True).init(
        full_config.number_of_good_contributors,
        full_config.number_of_bad_contributors,
        full_config.number_of_freerider_contributors,
        full_config.number_of_inactive_contributors,
        full_config.minimum_rounds,
        rpc,
        privkeys,
    )

    # Initialize on-chain GIR to 1 WAD for all users. This mirrors the
    # Solidity prior (WAD for new users) and makes the chain authoritative
    # from the first task onward.
    manager.initialize_user_balances(all_users)

    q_weight = preset.q_weight
    tr_weight = preset.tr_weight
    gir_weight = preset.gir_weight
    experiment_name = preset.name

    for i, task in enumerate(tasks):
        task_type = get_task_type(task.dataset)

        # Capture the rep state that will inform the selection decision.
        pre_state = {
            u.address: {
                "tr":      u.task_rep.get(task_type, 0),
                "gir":     u.global_integrity_rep,
                "q":       u.q_value.get(task_type, 0),
                "balance": u.balance,
            }
            for u in all_users
        }

        # Mirror the contract's selection to predict participants.
        # All users still register; the contract makes the final choice.
        log("multirep", f"\n=== Task {i+1}/{len(tasks)}: {task.dataset} (mode={training_mode.value}) ===")
        selected_users = getTopN(all_users, task.number_of_participants, task_type, q_weight, tr_weight, gir_weight)
        scores = {u.address: compute_user_score(u, task_type, q_weight, tr_weight, gir_weight) for u in all_users}

        # Compute Q updates now, before the run, so q_old is the pre-run value.
        # Applied after chain sync to ensure our Python-side Q is authoritative.
        q_updates = _compute_q_updates(all_users, selected_users, task_type)

        # Build ExperimentConfiguration from ONLY the selected users' specs so
        # contributor counts and the experiment fingerprint are correct.
        filtered_partitions = filter_partitions_for_users(selected_users)
        exp_config = task.to_experiment_config(partition_file)
        task.training_mode = training_mode

        exp_config.q_weight = q_weight
        exp_config.tr_weight = tr_weight
        exp_config.gir_weight = gir_weight
        fingerprint = exp_config.get_finger_print(selected_users)
        log("multirep", f"Run {i+1}/{len(tasks)} | Fall back runs {fallback_count} | dataset={task.dataset} | fp={fingerprint[:8]}...")

        # Create a folder for this task's output files.
        safe_dataset = task.dataset.replace("-", "_").replace(".", "_").lower()
        task_dir = tasks_dir / f"task_{i+1:03d}_{safe_dataset}_{fingerprint[:8]}"
        task_dir.mkdir(parents=True, exist_ok=True)

        cached_run = _fetch_cached_run(fingerprint)
        if cached_run is not None:
            log("multirep", f"Fingerprint {fingerprint[:8]}... found in RunRepo — skipping experiment.")
            _apply_cached_reps(all_users, cached_run, task_type)
            _apply_q_updates(all_users, q_updates, task_type)
            log("multirep", f"\n--- Reputation snapshot after task {i+1} (cached) ---")
            log_user_reputations(all_users, task_type, selected_users, q_weight, tr_weight, gir_weight)
            multirep_logger.log_task(
                task_index=i, dataset=task.dataset, task_type=task_type,
                fingerprint=fingerprint, was_cached=True,
                users=all_users, selected_users=selected_users,
                pre_state=pre_state, scores=scores,
            )
            continue

        # ------------------------------------------------------------------ #
        # Set up per-task CSV + PKL logging (always, for every live task).   #
        # For REMOTE replay the tarball files will overwrite the locals.     #
        # ------------------------------------------------------------------ #
        csv_name = (
            f"{task.dataset}-{exp_config.contribution_score_strategy}-"
            f"{exp_config.freerider_start_round}-{exp_config.freerider_noise_scale}-"
            f"{exp_config.malicious_start_round}-{exp_config.malicious_noise_scale}-"
            f"{exp_config.use_outlier_detection}-{{{uuid.uuid4()}}}.csv"
        )
        task_csv_path = task_dir / csv_name
        writer = AsyncWriter(task_csv_path, _OUTPUTHEADERS, _WRITERBUFFERSIZE, exp_config, "sample")
        task_logger = ExperimentLogger(experiment_id=task_csv_path.stem, metadata=vars(exp_config))
        # Point repo_dir at task_dir so the JSON run-repo file lands there.
        # For REMOTE mode this is overridden by extract_and_register_runrepo,
        # but the fallback path resets it (see _run_remote).
        fl_globals.repo_dir = str(task_dir)

        # ------------------------------------------------------------------ #
        # Dispatch: LOCAL / REMOTE                                            #
        # ------------------------------------------------------------------ #
        fl_globals.expected_fingerprint = fingerprint
        run_result = _run_preset(
            task, exp_config, all_users, manager, fingerprint, experiment_name,
            writer=writer, logger=task_logger, path=task_csv_path,
        )
        if run_result is None:
            writer.finish()
            multirep_logger.log_task(
                task_index=i, dataset=task.dataset, task_type=task_type,
                fingerprint=fingerprint, was_cached=False,
                users=all_users, selected_users=selected_users,
                pre_state=pre_state, scores=scores,
            )
            continue
        run_data, filename = run_result
        is_replay = isinstance(run_data, list)

        # Count every REMOTE→LOCAL fallback in one place.  Both the exception
        # path and the silent HardPlayBack fingerprint-mismatch path return a
        # non-replay result from a REMOTE-mode call.
        if training_mode == TrainingMode.REMOTE and not is_replay:
            fallback_count += 1

        # ------------------------------------------------------------------ #
        # Save per-task output files (CSV + PKL always; JSON via repo_dir)   #
        # ------------------------------------------------------------------ #

        # Always stop the writer thread and save the logger.  For replay runs
        # the writer will have written only the header; the real CSV/PKL are
        # in the tarball and copied below, overwriting the placeholder files.
        writer.finish()
        pkl_path = task_csv_path.with_suffix(".pkl")
        task_logger.save(pkl_path)
        task_pkl_path = pkl_path

        remote_src = Path(fl_globals.repo_dir)
        if remote_src.is_dir() and remote_src != task_dir:
            if is_replay:
                # Copy the real csv/pkl/json from the downloaded tarball into
                # task_dir, overwriting the placeholder files created above.
                copy_remote_task_files(remote_src, task_dir)
                # Prefer the copied PKL (from remote server) over the empty local one.
                copied_pkls = [p for p in task_dir.glob("*.pkl") if p != task_pkl_path]
                if copied_pkls:
                    task_pkl_path = copied_pkls[0]
            else:
                # HardPlayBack fingerprint-mismatch fallback: local training ran
                # with repo_dir pointing at the extraction dir, so the JSON landed
                # there. Copy it into task_dir.
                copy_remote_task_files(remote_src, task_dir)

        # Upload result to the API only for LOCAL runs; REMOTE results
        # either originated on the server or are already stored there.
        if training_mode == TrainingMode.LOCAL:
            from experiment.multirep.remote_client import _config_to_json_element
            _upload_run(fingerprint, filename, json.dumps(_config_to_json_element(exp_config)))

        # ------------------------------------------------------------------ #
        # Rep updates (chain-authoritative)                                   #
        # ------------------------------------------------------------------ #
        addresses = [u.address for u in all_users]
        if is_replay:
            _apply_trs_reps(all_users, run_data, task_type, manager, exp_config.reward)
        else:
            _sync_balances_from_challenge(all_users, run_data, manager)
        try:
            reps = manager.contract.functions.getGrsAndTrsBatch(addresses, task_type).call()
            update_users_from_reps(all_users, reps, task_type)
        except Exception as e:
            log("multirep", f"[warn] getGrsAndTrsBatch failed: {e}")
        _apply_q_updates(all_users, q_updates, task_type)

        # ------------------------------------------------------------------ #
        # Compute confidence + internals for session logging                  #
        # ------------------------------------------------------------------ #
        post_confidence, post_k, post_mean, post_m2 = _rep_internals_after_task(all_users, manager, task_type)

        # Load the task's tables for the session pickle (self-contained).
        task_run_data = load_task_pkl_tables(task_pkl_path) if task_pkl_path else None

        rel_pkl = (
            str(task_pkl_path.relative_to(session_dir)) if task_pkl_path else None
        )
        multirep_logger.log_task(
            task_index=i, dataset=task.dataset, task_type=task_type,
            fingerprint=fingerprint, was_cached=False,
            users=all_users, selected_users=selected_users,
            pre_state=pre_state, scores=scores,
            post_confidence=post_confidence, post_k=post_k,
            post_running_mean=post_mean, post_m2=post_m2,
            pkl_path=rel_pkl, run_data=task_run_data,
        )

        log("multirep", f"\n--- Reputation snapshot after task {i+1} ---")
        log_user_reputations(all_users, task_type, selected_users, q_weight)

    # ---------------------------------------------------------------------- #
    # Finalise session: save session.pkl + tarball                            #
    # ---------------------------------------------------------------------- #
    session_pkl = session_dir / "session.pkl"
    multirep_logger.save(session_pkl)
    log("multirep", f"Session pickle saved: {session_pkl}")

    try:
        tarball = pack_session_tarball(session_dir)
        log("multirep", f"Session tarball: {tarball}")
    except Exception as e:
        log("multirep", f"[warn] Tarball creation failed: {e}")

    log("multirep", "\n=== All tasks complete. ===")
    log("multirep", f"Fall back runs: {fallback_count}")
    log("multirep", f"Session data: {session_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing as mp
    if False:
        mp.freeze_support()
    main()
    for p in mp.active_children():
        p.terminate()
    print("Done :)")
