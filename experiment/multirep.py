import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from typing import List
import requests
import experiment.experiment_runner as ExperimentRunner
from experiment.experiment_runner import setup_connection
from experiment.multirep.MultirepRunConfig import MultirepRunConfig
from openfl.contracts import FLManager as Manager
from openfl.ml.partition_spec import ANY_DATASET, load_dataset_partition_specs
from openfl.utils.types.User import User
from openfl.utils.printer import log, set_log_file
from openfl.utils.W3Helper import get_PRIVKEYS, get_RPC_Endpoint


# ---------------------------------------------------------------------------
# Scoring configuration — mirrors the smart contract's getTopN selection logic.
# Used to predict participant selection for fingerprinting / RunRepo caching.
# ---------------------------------------------------------------------------

# "q_weighted"      → score = max(1, q) * (task_rep*0.6 + gir*0.4)   [mirrors on-chain getTopN]
# "reputation_only" → score =              task_rep*0.6 + gir*0.4
SCORING_MODE = "q_weighted"

TASK_REP_WEIGHT = 0.6
GLOBAL_REP_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Presets — fill in before running
# ---------------------------------------------------------------------------

presets: List[MultirepRunConfig] = [
    MultirepRunConfig(
        partition_file="experiment/partitions/example.json",
        dataset="MNIST",
        minimum_rounds=5,
        number_of_participants=6,
    ),
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def compute_user_score(user: User, task_type: int) -> float:
    base = (
        user.task_rep.get(task_type, 0) * TASK_REP_WEIGHT
        + user.global_integrity_rep * GLOBAL_REP_WEIGHT
    )
    if SCORING_MODE == "q_weighted":
        q = max(1, user.q_value.get(task_type, 0))
        return q * base
    return base


def getTopN(users: List[User], n: int, task_type: int) -> List[User]:
    """Mirror the smart contract's participant selection for fingerprinting."""
    scores = [(compute_user_score(u, task_type), u) for u in users]
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [u for _, u in scores[:n]]
    selected_set = {u.address for u in selected}
    log("multirep", f"Selection (top {n} of {len(users)}, task_type={task_type}, mode={SCORING_MODE}):")
    for score, u in scores:
        marker = "SELECTED" if u.address in selected_set else "       -"
        log("multirep", f"  [{marker}]  score={score:>12.4f}  User {u.number}")
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
    for rep in reps:
        address, task_rep, global_integrity_rep, total_contrib_score, q_value = rep
        user = users_by_address.get(address.lower())
        if user is None:
            continue
        user.task_rep[task_type] = task_rep
        user.global_integrity_rep = global_integrity_rep
        user.total_contrib_score = total_contrib_score
        user.q_value[task_type] = q_value


# ---------------------------------------------------------------------------
# Partition filtering
# ---------------------------------------------------------------------------

def log_user_reputations(users: List[User], task_type: int, selected_users: List[User]) -> None:
    """Log reputation fields and selection status for every user."""
    selected_set = {u.address for u in selected_users}
    log("multirep", "─" * 88)
    log("multirep", f"{'User':<8} {'Address':<20}  {'TaskRep':>14} {'GIR':>14} {'Q':>10} {'Score':>12}  {'Selected':>8}")
    log("multirep", "─" * 88)
    for u in users:
        score = compute_user_score(u, task_type)
        tr = u.task_rep.get(task_type, 0)
        gir = u.global_integrity_rep
        q = u.q_value.get(task_type, 1)
        selected = "YES" if u.address in selected_set else "no"
        label = f"User {u.number}"
        addr = u.address[:20] if u.address else "N/A"
        log("multirep", f"{label:<8} {addr:<20}  {tr:>14} {gir:>14} {q:>10.4f} {score:>12.4f}  {selected:>8}")
    log("multirep", "─" * 88)


def filter_partitions_for_users(selected_users: List[User]) -> dict:
    """Build a {ANY_DATASET: {user_index: UserPartitionSpec}} dict from selected users.

    The ANY_DATASET key lets ExperimentConfiguration.get_partition_specs() find
    these specs regardless of the active dataset name.
    """
    specs = {}
    for user in selected_users:
        if user.partition_spec is not None:
            specs[user.partition_spec.user_index] = user.partition_spec
    return {ANY_DATASET: specs}


# ---------------------------------------------------------------------------
# RunRepo cache lookup
# ---------------------------------------------------------------------------

def _apply_cached_reps(users: List[User], cached_run: dict, task_type: int) -> None:
    """Apply reputation data from a cached API run response to user objects."""
    reps_data = cached_run.get("reputations", [])
    if not reps_data:
        log("multirep", "[warn] No reputation data in cached run — rep state unchanged.")
        return
    reps = [
        (r["address"], r["taskRep"], r["globalIntegrityRep"], r["totalContribScore"], r["qValue"])
        for r in reps_data
    ]
    update_users_from_reps(users, reps, task_type)


def _fetch_cached_run(fingerprint: str):
    """Return API run data if a cached result exists for this fingerprint, else None."""
    api_url = os.environ.get("API_URL")
    if not api_url:
        return None
    try:
        res = requests.get(f"{api_url}/runs/by-fingerprint/{fingerprint}", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        log("multirep", f"[warn] fingerprint cache lookup failed: {e}")
    return None


def _register_run(api_url: str, fingerprint: str, config: str) -> str | None:
    try:
        res = requests.post(f"{api_url}/runs", json={"fingerprint": fingerprint, "config": config}, timeout=10)
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
# Main
# ---------------------------------------------------------------------------

def main():
    if not presets:
        log("multirep", "No presets configured — nothing to run.")
        return

    # Set up persistent log file for this session.
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_file(str(log_dir / f"multirep_{session_ts}.log"))
    log("multirep", f"=== MultiRep session started {session_ts} ===")

    first_preset = presets[0]

    # Load ALL partition specs from the JSON once.  Users are created from
    # this full pool and keep their data partitions for the entire session.
    full_config = first_preset.to_experiment_config()
    all_users = ExperimentRunner.build_users(full_config)

    # Deploy the manager contract once before the loop so it persists across
    # all presets, including those that are skipped via the RunRepo cache.
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

    for i, preset in enumerate(presets):
        task_type = get_task_type(preset.dataset)

        # Mirror the contract's selection to predict participants.
        # All users still register; the contract makes the final choice.
        log("multirep", f"\n=== Preset {i+1}/{len(presets)}: {preset.dataset} ===")
        selected_users = getTopN(all_users, preset.number_of_participants, task_type)

        # Build ExperimentConfiguration from ONLY the selected users' specs so
        # contributor counts and the experiment fingerprint are correct.
        filtered_partitions = filter_partitions_for_users(selected_users)
        exp_config = preset.to_experiment_config_with_partitions(filtered_partitions)

        fingerprint = exp_config.get_finger_print(selected_users)
        log("multirep", f"Run {i+1}/{len(presets)} | dataset={preset.dataset} | fp={fingerprint[:8]}...")

        cached_run = _fetch_cached_run(fingerprint)
        if cached_run is not None:
            log("multirep", f"Fingerprint {fingerprint[:8]}... found in RunRepo — skipping experiment.")
            _apply_cached_reps(all_users, cached_run, task_type)
            log("multirep", f"\n--- Reputation snapshot after preset {i+1} (cached) ---")
            log_user_reputations(all_users, task_type, selected_users)
            continue

        # All users register; experiment_runner handles data re-partitioning
        # (stable because user properties + seed are fixed) and manager reuse.
        result, filename = ExperimentRunner.run_experiment(
            preset.dataset,
            exp_config,
            prebuilt_users=all_users,
            prebuilt_manager=manager,
        )

        _upload_run(fingerprint, filename, exp_config)

        # Sync reputations from chain so the next preset's getTopN is current.
        addresses = [u.address for u in all_users]
        try:
            reps = manager.contract.functions.getGrsAndTrsBatch(addresses, task_type).call()
            update_users_from_reps(all_users, reps, task_type)
        except Exception as e:
            log("multirep", f"[warn] getGrsAndTrsBatch failed: {e}")

        log("multirep", f"\n--- Reputation snapshot after preset {i+1} ---")
        log_user_reputations(all_users, task_type, selected_users)

    log("multirep", "\n=== All presets complete. ===")


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
