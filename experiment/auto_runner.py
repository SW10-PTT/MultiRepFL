from pathlib import Path
import gc
import pprint
import socket
import sys
import threading

from openfl.utils.require_env import require_env_var
from openfl.utils.types.User import User
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from openfl.utils.printer import log

API = require_env_var("API_URL")

worker_id = None

def register_worker():
    res = requests.post(f"{API}/workers/register", json={
        "name": socket.gethostname()[:6]
    })
    res.raise_for_status()
    return res.json()["workerId"]

def claim_run(worker_id):
    res = requests.post(f"{API}/runs/claim", json={
        "workerId": worker_id
    })

    if res.status_code != 200:
        return None

    return res.json() 

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
        requests.post(f"{API}/workers/heartbeat", json={
            "workerId": worker_id,
            "progress": globals.progress
        }, timeout=2)
    except Exception:
        # ignore failures (network hiccups etc.)
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

            config = coerce_types(config)
            #pprint.pp(config)
            config = ExperimentConfiguration(**config)

            start_heartbeat_loop()

            startTime = datetime.now().strftime("%d-%m-%y--%H_%M_%S")
            path = getPath(config, startTime, config.dataset, RESULTDATAFOLDER)

            globals.repo_dir = path.parent

            writer = AsyncWriter(path, OUTPUTHEADERS, WRITERBUFFERSIZE, config, "sample")
            logger = ExperimentLogger(experiment_id=path.stem, metadata=vars(config))

            users = build_users(config)
            (experiment, filename) = experiment_runner.run_experiment(
                config.dataset, config, writer, logger, path,
                prebuilt_users=users,
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

            user_guids = [
                {"guid": u.guid, "address": u.address}
                for u in users if u.guid is not None
            ]
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

heartbeat_stop = None
heartbeat_thread = None

def reset(experiment, filename):
    experiment.py
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
    globals.reuse_runs = globals.ReplayMode.Record
    worker_loop()

if __name__ == "__main__":
    main()