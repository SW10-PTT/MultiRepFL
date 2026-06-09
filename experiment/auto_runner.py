from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))

import gc
import os
import platform
import pprint
import socket
import subprocess
import threading

from openfl.utils.require_env import require_env_var
from openfl.utils.types.User import User

import tarfile
import tempfile

from experiment.experiment_configuration import ExperimentConfiguration
from experiment.print_config import AGGRESSIVE_GC
import requests
import time
import json
from datetime import datetime

from analysis import ExperimentLogger
from experiment import experiment_runner
from experiment.experiment_runner import build_users
from experiment.helper import getPath
from openfl.utils.async_writer import AsyncWriter
from openfl.api import globals

from openfl.utils import printer, config
from openfl.utils.printer import log, set_log_file

API = require_env_var("API_URL")

worker_id = None

_RAM_THRESHOLD_GB = 10.0

def _free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise OSError("Cannot determine available RAM")

def _check_ram_startup() -> None:
    """Startup guard: exit immediately if free RAM < threshold. No restart."""
    try:
        free_gb = _free_ram_gb()
    except Exception as e:
        log("autorunner", f"[mem] RAM check failed: {e} — skipping")
        return
    if free_gb < _RAM_THRESHOLD_GB:
        log("autorunner", f"[mem] Only {free_gb:.1f} GB free (need {_RAM_THRESHOLD_GB} GB). Exiting.")
        sys.exit(1)

def _check_ram_and_maybe_restart() -> None:
    """Post-run check: if free RAM < threshold, restart after 30 s."""
    try:
        free_gb = _free_ram_gb()
    except Exception as e:
        log("autorunner", f"[mem] RAM check failed: {e} — skipping")
        return
    if free_gb >= _RAM_THRESHOLD_GB:
        return
    log("autorunner", f"[mem] Only {free_gb:.1f} GB free (need {_RAM_THRESHOLD_GB} GB). Restarting in 30 s...")
    stop_heartbeat_loop()
    time.sleep(30)
    os.environ["_AUTORUNNER_RESTART_REASON"] = f"low_ram:{free_gb:.1f}GB"
    _restart()

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

def _restart() -> None:
    """Stop blockchain, then replace this process with a fresh instance."""
    try:
        from experiment.blockchain_launcher import _cleanup as _bc_cleanup
        _bc_cleanup()
    except Exception:
        pass
    args = [sys.executable] + sys.argv
    if platform.system() == "Windows":
        subprocess.Popen(args)
        os._exit(0)  # Hard exit — terminates all threads, safe from non-main thread
    else:
        os.execv(sys.executable, args)

def _switch_to_commit(sha: str) -> None:
    log("autorunner", f"[version] Server requires commit {sha[:8]}. Fetching and switching...")
    subprocess.check_call(["git", "fetch", "origin"], cwd=str(_repo_root))
    subprocess.check_call(["git", "checkout", "-f", sha], cwd=str(_repo_root))
    log("autorunner", "[version] Compiling contracts...")
    subprocess.check_call([sys.executable, "scripts/compile_contracts.py"], cwd=str(_repo_root))
    log("autorunner", f"[version] Done. Restarting as {sha[:8]}...")
    os.environ["_AUTORUNNER_RESTART_REASON"] = f"version_switch:{sha[:8]}"
    _restart()

def register_worker():
    res = requests.post(f"{API}/workers/register", json={
        "name": socket.gethostname()[:6],
        "gitCommit": _git_commit(),
    })
    if res.status_code == 409:
        body = res.json()
        if body.get("denied") and body.get("expectedCommit"):
            _switch_to_commit(body["expectedCommit"])
    res.raise_for_status()
    data = res.json()
    if data.get("shutdown"):
        log("autorunner", "Server requested shutdown. Exiting.")
        raise SystemExit(0)
    return data["workerId"]

def claim_run(worker_id):
    res = requests.post(f"{API}/runs/claim", json={
        "workerId": worker_id
    })

    if res.status_code == 409:
        body = res.json()
        if body.get("denied") and body.get("expectedCommit"):
            _switch_to_commit(body["expectedCommit"])
        return None

    if res.status_code != 200:
        return None

    data = res.json()
    if data.get("shutdown"):
        log("autorunner", "Server requested shutdown. Exiting.")
        try:
            from experiment.blockchain_launcher import _cleanup as _bc_cleanup
            _bc_cleanup()
        except Exception:
            pass
        os._exit(0)
    return data

def get_upload_url(run_id, fingerprint):
    res = requests.post(
        f"{API}/runs/{run_id}/upload-url",
        json=fingerprint.name
    )

    res.raise_for_status()

    return res.json()

def heartbeat(worker_id):
    from openfl.api import globals
    try:
        res = requests.post(f"{API}/workers/heartbeat", json={
            "workerId": worker_id,
            "progress": globals.progress
        }, timeout=2)
        if res.status_code == 409:
            body = res.json()
            if body.get("denied") and body.get("expectedCommit"):
                _switch_to_commit(body["expectedCommit"])
        elif res.ok and res.json().get("shutdown"):
            log("autorunner", "Server requested shutdown. Exiting.")
            try:
                from experiment.blockchain_launcher import _cleanup as _bc_cleanup
                _bc_cleanup()
            except Exception:
                pass
            os._exit(0)
    except Exception:
        pass

def create_tarball(folder_path: Path, fingerpint):
    folder_path_actual = folder_path.parent
    archive_path = folder_path_actual.parent / f"{fingerpint.stem}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(folder_path_actual, folder_path.stem)

    return archive_path


def upload_file(upload_url, file_path: Path):
    log("autorunner","Uploading:", file_path)

    with open(file_path, "rb") as f:
        res = requests.put(
            upload_url,
            data=f,
            headers={
                "Content-Type": "application/gzip"
            },
            timeout=300
        )

    log("autorunner", res.status_code)
    log("autorunner", res.text)

    res.raise_for_status()

def complete_run(run_id, user_guids=None):
    body = {"RunId": str(run_id)}
    if user_guids:
        body["userGuids"] = user_guids
    response = requests.post(f"{API}/runs/{run_id}/complete", json=body)
    response.raise_for_status()

def fail_run(run_id, error):
    requests.post(f"{API}/runs/{run_id}/fail", json={
        "error": str(error)
    })

def check_worker_exists():
    if worker_id == None:
        return False
    res = requests.get(f"{API}/workers/{worker_id}")
    return res.status_code == 200

OUTPUTHEADERS = [
    "round",
    "time",
    "globalAcc",
    "globalLoss",
    "GRS",
    "accAvgPerUser",
    "lossAvgPerUser",
    "rewards",
    "conctractBalanceRewards",
    "punishments",
    "contributionScores",
    "feedbackMatrix",
    "disqualifiedUsers",
    "userStatuses",
    "GasTransactions",
    "Contrib"
    ]
WRITERBUFFERSIZE = 200

RESULTDATAFOLDER = Path(__file__).resolve().parent.joinpath("data/experimentData")

def coerce_types(value):
    if isinstance(value, dict):
        return {k: coerce_types(v) for k, v in value.items()}

    if isinstance(value, list):
        return [coerce_types(v) for v in value]

    if isinstance(value, str):
        v = value.strip()

        # None
        if v.lower() in ["none", "null"]:
            return None

        # bool
        if v.lower() in ["true", "false"]:
            return v.lower() == "true"

        # int
        try:
            if "." not in v:
                return int(v)
        except Exception:
            pass

        # float
        try:
            f = float(v)
            if f.is_integer():
                return int(f)
            return f
        except Exception:
            pass

        return v

    return value

def registerWorkerLoop():
    global worker_id
    while True:
        try:
            worker_id = register_worker()
            log("autorunner", "Worker registered:", worker_id)
            break
        except Exception:
            log("autorunner", "Failed to register worker, trying again in 10 seconds...")
            time.sleep(10)
            continue

def worker_loop():
    global worker_id
    
    while True:
        experiment = None
        writer = None
        logger = None
        experiment, filename = None, None
        try:
            if not check_worker_exists():
                registerWorkerLoop()

            run = claim_run(worker_id)

            if not run:
                log("autorunner", "No runs available...")
                time.sleep(5)
                continue

            run_id = run["id"]

            log("autorunner", "Running:", run_id)

            config = run["config"]

            if isinstance(config, str):
                config = json.loads(config)

            # Pop custom multirep fields before coerce_types so the dict is clean.
            assert isinstance(config, dict)
            expected_fingerprint = config.pop("expectedFingerprint", None)
            initial_rep_state_by_guid: dict = config.pop("initialRepState", None) or {}
            config = coerce_types(config)
            assert isinstance(config, dict)
            config = ExperimentConfiguration(**config)

            start_heartbeat_loop()

            startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")
            path = getPath(config, startTime, config.dataset, RESULTDATAFOLDER)


            globals.repo_dir = path.parent

            writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
            logger = ExperimentLogger(experiment_id=path.stem, metadata=vars(config))

            users = build_users(config)
            # Address maps for THIS machine's blockchain.
            # Guids match multirep's users; addresses do not — keep these maps separate.
            addr_to_id: dict[str, str] = {u.address.lower(): u.guid for u in users if u.guid}
            id_to_addr: dict[str, str] = {u.guid: u.address for u in users if u.guid}

            # Convert guid-keyed rep state to local addresses.
            initial_rep_state = {
                id_to_addr[guid]: {k: int(v) for k, v in state.items() if k != "task_type"}
                for guid, state in initial_rep_state_by_guid.items()
                if guid in id_to_addr
            }
            missing = [g for g in initial_rep_state_by_guid if g not in id_to_addr]
            log("autorunner", f"[rep_state] run={run_id} received state for {len(initial_rep_state_by_guid)} users, mapped {len(initial_rep_state)}, unmatched_guids={missing}")

            if expected_fingerprint is not None:
                actual_fingerprint = config.get_finger_print(users)
                if actual_fingerprint != expected_fingerprint:
                    raise ValueError(
                        f"Fingerprint mismatch before run: "
                        f"expected={expected_fingerprint[:8]}... "
                        f"actual={actual_fingerprint[:8]}... — "
                        f"participant selection differs between multirep and auto_runner"
                    )

            (experiment, filename) = experiment_runner.run_experiment(
                config.dataset, config, writer, logger, path,
                prebuilt_users=users,
                initial_rep_state=initial_rep_state or None,
            )

            writer.finish()
            logger.save(path.with_suffix(".pkl"))

            # upload result
            upload_info = get_upload_url(run_id, filename)

            archive_path = create_tarball(path, filename)

            upload_file(
                upload_info["uploadUrl"],
                archive_path
            )

            user_guids = [{"Guid": u.guid, "Address": u.address} for u in users if u.guid is not None]
            complete_run(run_id, user_guids)
            reset(experiment, filename)
            #stop_heartbeat_loop()

        except Exception as e:
            stop_heartbeat_loop()
            log("autorunner", "Run failed:", e)
            try:
                fail_run(run_id, e)
                reset(experiment, filename)
            except Exception:
                reset(experiment, filename)
                time.sleep(10)
        finally:
            if AGGRESSIVE_GC:
                # Drop the prior run's PytorchModel + DataLoaders before claiming
                # the next run, so persistent DataLoader workers (and their FDs)
                # are reclaimed instead of accumulating across iterations.
                del experiment, writer, logger
                gc.collect()

        _check_ram_and_maybe_restart()

heartbeat_stop = None
heartbeat_thread = None

def reset(experiment, filename):
    del experiment
    del filename
    globals.progress = 0
    User.user_count = 0

def start_heartbeat_loop():
    global heartbeat_stop, heartbeat_thread

    stop_event = threading.Event()
    heartbeat_stop = stop_event

    def loop():
        while not stop_event.is_set():
            if worker_id:
                heartbeat(worker_id)

            time.sleep(10)

    heartbeat_thread = threading.Thread(
        target=loop,
        daemon=True
    )

    heartbeat_thread.start()

def stop_heartbeat_loop():
    global heartbeat_stop, heartbeat_thread

    if heartbeat_stop:
        heartbeat_stop.set()

    if heartbeat_thread:
        heartbeat_thread.join(timeout=1)

    heartbeat_stop = None
    heartbeat_thread = None

def main():
    log_dir = Path(__file__).resolve().parent / "data" / "logs"
    log_dir.mkdir(exist_ok=True)
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_file(str(log_dir / f"autorunner_{session_ts}.log"))
    restart_reason = os.environ.pop("_AUTORUNNER_RESTART_REASON", None)
    if restart_reason:
        log("autorunner", f"[mem] Restarted — previous process shut down due to: {restart_reason}")
    globals.reuse_runs = globals.ReplayMode.Record
    worker_loop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the auto_runner FL experiment worker.")
    blockchain_group = parser.add_mutually_exclusive_group()
    blockchain_group.add_argument(
        "--anvil", action="store_true",
        help="Start a local Anvil node (30 accounts) and use it as the RPC endpoint.",
    )
    blockchain_group.add_argument(
        "--ganache", action="store_true",
        help="Start a local Ganache node (30 accounts) and use it as the RPC endpoint.",
    )
    args = parser.parse_args()

    _check_ram_startup()

    if args.anvil or args.ganache:
        from experiment.blockchain_launcher import start as _start_blockchain, _cleanup as _bc_cleanup
        _start_blockchain("anvil" if args.anvil else "ganache")
    else:
        _bc_cleanup = None

    try:
        main()
    finally:
        if _bc_cleanup is not None:
            _bc_cleanup()