import json
import os
import sys
import tarfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from typing import List
import requests
import experiment.experiment_runner as ExperimentRunner
from experiment.experiment_runner import setup_connection
from experiment.multirep.MultirepPreset import MultirepPreset
from experiment.multirep.MultirepRunConfig import MultirepRunConfig
from experiment.multirep.training_mode import TrainingMode
from openfl.contracts import FLManager as Manager
from openfl.utils.types.User import User
from openfl.utils.printer import log, set_log_file, set_enabled_tags
from openfl.utils.W3Helper import get_PRIVKEYS, get_RPC_Endpoint
from openfl.api import globals as fl_globals
from openfl.api.globals import ReplayMode


# ---------------------------------------------------------------------------
# Scoring configuration — mirrors the smart contract's getTopN selection logic.
# Used to predict participant selection for fingerprinting / RunRepo caching.
# ---------------------------------------------------------------------------

_WAD = 10 ** 18  # all on-chain rep values are WAD-scaled


# ---------------------------------------------------------------------------
# Preset file — fill in before running
# ---------------------------------------------------------------------------

preset_file = "experiment/presets/example.json"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_user_score(user: User, task_type: int, q_weight: float = 0.0, tr_weight: int = 6, gir_weight: int = 4) -> int:
    # score = (taskRep * tr_weight + gir * gir_weight) / (tr_weight + gir_weight) + q_weight * q * WAD
    denom = tr_weight + gir_weight
    base = user.task_rep.get(task_type, 0) * tr_weight + user.global_integrity_rep * gir_weight
    normal_weight = base // denom
    q = user.q_value.get(task_type, 0.0)
    return int(normal_weight + q_weight * q * _WAD)


def getTopN(users: List[User], n: int, task_type: int, q_weight: float = 0.0, tr_weight: int = 6, gir_weight: int = 4) -> List[User]:
    """Mirror the smart contract's participant selection for fingerprinting."""
    scores = [(compute_user_score(u, task_type, q_weight, tr_weight, gir_weight), u) for u in users]
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [u for _, u in scores[:n]]
    selected_set = {u.address for u in selected}
    log("multirep", f"Selection (top {n} of {len(users)}, task_type={task_type}):")
    for score, u in scores:
        marker = "SELECTED" if u.address in selected_set else "       -"
        name = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else f"User {u.number}"
        log("multirep", f"  [{marker}]  score={score / _WAD:>8.4f}  {name}")
    return selected


# ---------------------------------------------------------------------------
# On-chain reputation helpers
# ---------------------------------------------------------------------------

def get_task_type(dataset: str) -> int:
    """Map dataset name to TaskType enum value from Types.sol."""
    d = dataset.lower()
    if "mnist" in d:
        return 6  # Images_MNIST
    if "cifar" in d:
        return 7  # Images_CIFAR10
    raise ValueError(f"Unknown task type for dataset: {dataset!r}")


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
            guid = rep.get("guid")
        else:
            address, task_rep, global_integrity_rep, total_contrib_score, q_value = rep
            guid = None
        user = (users_by_guid.get(guid) if guid else None) or users_by_address.get(address.lower())
        if user is None:
            continue
        user.task_rep[task_type] = task_rep
        user.global_integrity_rep = global_integrity_rep
        user.total_contrib_score = total_contrib_score
        user.q_value[task_type] = q_value


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
        q_old = float(user.q_value.get(task_type, 0.0))
        is_selected = user.address in selected_addrs or (
            user.guid is not None and user.guid in selected_guids
        )
        delta = ratio - (1.0 if is_selected else 0.0)
        updates[user.address] = max(0.0, q_old + delta)
    return updates


def _apply_q_updates(users: List[User], q_updates: dict, task_type: int) -> None:
    for user in users:
        if user.address in q_updates:
            user.q_value[task_type] = q_updates[user.address]


# ---------------------------------------------------------------------------
# TRS (replay rep update)
# ---------------------------------------------------------------------------

def _apply_trs_reps(users: List[User], trs: list, task_type: int, manager) -> None:
    """Apply task-rep delta + delta_balance on-chain and in Python for replayed runs.

    trs format: (guid, delta_task_rep, delta_balance, pos_votes, total_votes).
    Both deltas are applied additively (clamped to 0) and written to the manager
    contract so the chain remains the authoritative source of truth.
    """
    users_by_guid = {u.guid: u for u in users if u.guid is not None}
    for entry in trs:
        guid, delta, delta_balance = str(entry[0]), entry[1], entry[2]
        user = users_by_guid.get(guid)
        if user is None:
            continue
        new_task_rep = max(0, user.task_rep.get(task_type, 0) + delta)
        new_gir = max(0, user.global_integrity_rep + delta_balance)
        manager.set_user_task_rep(user.address, task_type, new_task_rep)
        manager.set_user_integrity_rep(user.address, new_gir)
        user.task_rep[task_type] = new_task_rep
        user.global_integrity_rep = new_gir


# ---------------------------------------------------------------------------
# Partition filtering
# ---------------------------------------------------------------------------

def log_user_reputations(users: List[User], task_type: int, selected_users: List[User], q_weight: float = 0.0, tr_weight: int = 6, gir_weight: int = 4) -> None:
    """Log reputation fields and selection status for every user."""
    selected_set = {u.address for u in selected_users}
    log("multirep", "─" * 80)
    log("multirep", f"{'User':<12} {'Address':<20}  {'TaskRep':>8} {'Balance':>8} {'Q':>7} {'Score':>8}  {'Selected':>8}")
    log("multirep", "─" * 80)
    for u in users:
        score = compute_user_score(u, task_type, q_weight, tr_weight, gir_weight)
        tr = u.task_rep.get(task_type, 0) / _WAD
        balance = u.global_integrity_rep / _WAD
        q = float(u.q_value.get(task_type, 0.0))
        score_display = score / _WAD
        selected = "YES" if u.address in selected_set else "no"
        name = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else None
        label = name if name else f"User {u.number}"
        addr = u.address[:20] if u.address else "N/A"
        log("multirep", f"{label:<12} {addr:<20}  {tr:>8.4f} {balance:>8.4f} {q:>7.3f} {score_display:>8.4f}  {selected:>8}")
    log("multirep", "─" * 80)


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

def _run_preset(preset: MultirepRunConfig, exp_config, all_users, manager, fingerprint):
    from experiment.multirep.training_mode import TrainingMode

    if preset.training_mode == TrainingMode.REMOTE:
        return _run_remote(preset, exp_config, all_users, manager, fingerprint)

    # LOCAL
    fl_globals.reuse_runs = ReplayMode.Record
    return ExperimentRunner.run_experiment(
        preset.dataset,
        exp_config,
        prebuilt_users=all_users,
        prebuilt_manager=manager,
    )


def _run_remote(preset: MultirepRunConfig, exp_config, all_users, manager, fingerprint: str):
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
                name=exp_config.name or f"multirep-{preset.dataset}",
            )

        return ExperimentRunner.run_experiment(
            preset.dataset,
            exp_config,
            prebuilt_users=all_users,
            prebuilt_manager=manager,
        )

    except Exception as e:
        log("multirep", f"[REMOTE] Failed — falling back to LOCAL.\nReason: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        fl_globals.reuse_runs = ReplayMode.Record
        return ExperimentRunner.run_experiment(
            preset.dataset,
            exp_config,
            prebuilt_users=all_users,
            prebuilt_manager=manager,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    preset = MultirepPreset.from_file(preset_file)
    tasks = preset.tasks
    partition_file = preset.partition_file
    training_mode = preset.training_mode

    if not tasks:
        log("multirep", "No tasks configured — nothing to run.")
        return

    # Set up persistent log file for this session.
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_file(str(log_dir / f"multirep_{session_ts}.log"))

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

    for i, task in enumerate(tasks):
        task_type = get_task_type(task.dataset)

        # Mirror the contract's selection to predict participants.
        # All users still register; the contract makes the final choice.
        log("multirep", f"\n=== Task {i+1}/{len(tasks)}: {task.dataset} (mode={training_mode.value}) ===")
        selected_users = getTopN(all_users, task.number_of_participants, task_type, q_weight, tr_weight, gir_weight)

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
        log("multirep", f"Run {i+1}/{len(tasks)} | dataset={task.dataset} | fp={fingerprint[:8]}...")

        cached_run = _fetch_cached_run(fingerprint)
        if cached_run is not None:
            log("multirep", f"Fingerprint {fingerprint[:8]}... found in RunRepo — skipping experiment.")
            _apply_cached_reps(all_users, cached_run, task_type)
            _apply_q_updates(all_users, q_updates, task_type)
            log("multirep", f"\n--- Reputation snapshot after task {i+1} (cached) ---")
            log_user_reputations(all_users, task_type, selected_users, q_weight, tr_weight, gir_weight)
            continue

        # ------------------------------------------------------------------ #
        # Dispatch: LOCAL / REMOTE                                            #
        # ------------------------------------------------------------------ #
        fl_globals.expected_fingerprint = fingerprint
        run_result = _run_preset(task, exp_config, all_users, manager, fingerprint)
        if run_result is None:
            continue
        run_data, filename = run_result

        # Upload result to the API only for LOCAL runs; REMOTE results
        # either originated on the server or are already stored there.
        if training_mode == TrainingMode.LOCAL:
            _upload_run(fingerprint, filename, json.dumps(exp_config.to_dict()))

        # Sync reputations. The chain is authoritative; Python mirrors it.
        # For replay runs: first read pre-run chain state into Python, then
        # apply trs deltas on-chain (and Python-side), keeping both in sync.
        # For local runs: the challenge already updated the chain; just read it.
        # Q is computed Python-side and applied last.
        addresses = [u.address for u in all_users]
        is_replay = isinstance(run_data, list)
        try:
            reps = manager.contract.functions.getGrsAndTrsBatch(addresses, task_type).call()
            update_users_from_reps(all_users, reps, task_type)
        except Exception as e:
            log("multirep", f"[warn] getGrsAndTrsBatch failed: {e}")
        if is_replay:
            _apply_trs_reps(all_users, run_data, task_type, manager)
        _apply_q_updates(all_users, q_updates, task_type)

        log("multirep", f"\n--- Reputation snapshot after task {i+1} ---")
        log_user_reputations(all_users, task_type, selected_users, q_weight)

    log("multirep", "\n=== All tasks complete. ===")


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
