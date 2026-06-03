"""remote_client.py — helpers for submitting experiments to the remote API,
polling run status, and downloading the result tarball so it can be used as a
local RunRepo for replay.

Download flow
-------------
After a run reaches "Completed" status, the tarball is fetched via:

    POST /api/runs/{id}/upload-url   →  { "uploadUrl": "<presigned-url>" }

That presigned URL (Azure SAS / S3 pre-signed GET) is then streamed to disk.

The tarball contains one directory whose root holds the training_trace JSON
file.  After extraction, globals.repo_dir is pointed at that directory so the
normal PlayBack path in get_filename() finds and replays the trace.
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import requests

from openfl.utils.printer import log

# ---------------------------------------------------------------------------
# Status constants (mirror ExperimentRunStatus in the .NET API)
# ---------------------------------------------------------------------------

_TERMINAL_OK   = {"completed"}
_TERMINAL_FAIL = {"failed", "cancelled"}
_POLL_INTERVAL = 2   # seconds between status checks
_POLL_TIMEOUT  = 3600 # seconds before giving up (1 h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_url() -> str:
    url = os.environ.get("API_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("API_URL environment variable is not set")
    return url


def _config_to_json_element(experiment_config) -> Any:
    """Serialise ExperimentConfiguration to a plain dict that the API accepts
    as the ConfigJson field (a raw JsonElement on the .NET side).

    per_user_partitions contains UserPartitionSpec objects which are not JSON-
    serialisable by default, so we delegate to to_dict() which returns plain
    Python primitives, and rely on json.dumps for the final serialisation step.
    """
    raw = experiment_config.to_dict()

    # per_user_partitions inside ExperimentConfiguration is
    #   {dataset_key: {user_index: UserPartitionSpec}}
    # Serialise each spec while preserving the dataset-key layer so the remote
    # worker can reconstruct the exact same structure (and produce the same fingerprint).
    if raw.get("per_user_partitions"):
        raw["per_user_partitions"] = {
            dataset_key: {
                uk: (spec.serialize() if hasattr(spec, "serialize") else vars(spec))
                for uk, spec in specs.items()
            }
            for dataset_key, specs in raw["per_user_partitions"].items()
        }

    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_remote_experiment(
    experiment_config,
    fingerprint: str,
    name: str | None = None,
    wanted_runs: int = 1,
) -> str:
    """POST /custom-experiments/start and return the run_id string."""
    config_payload = _config_to_json_element(experiment_config)
    # Embed the expected fingerprint so auto_runner can validate before running.
    config_payload["expectedFingerprint"] = fingerprint

    body = {
        "wantedRuns": wanted_runs,
        "name": name,
        "configJson": config_payload,
    }

    url = f"{_api_url()}/custom-experiments/start"
    log("remote_client", f"Submitting remote experiment to {url} …")

    res = requests.post(url, json=body, timeout=30)
    res.raise_for_status()

    data = res.json()
    run_id = data["runId"]
    log("remote_client", f"Remote run submitted: runId={run_id}")
    return run_id


def poll_run_status(run_id: str, timeout: int = _POLL_TIMEOUT, interval: int = _POLL_INTERVAL) -> dict:
    """Poll GET /runs/:runId until terminal status; return the final response dict.

    Raises TimeoutError if *timeout* seconds elapse without a terminal status.
    Raises RuntimeError if the run enters a failed/cancelled state.
    """
    url = f"{_api_url()}/runs/{run_id}"
    deadline = time.monotonic() + timeout

    while True:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        data = res.json()

        status = data.get("status", "").lower()
        log("remote_client", f"Run {run_id} — status: {status}")

        if status in _TERMINAL_OK:
            return data

        if status in _TERMINAL_FAIL:
            raise RuntimeError(f"Remote run {run_id} ended with status '{status}'")

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Remote run {run_id} did not complete within {timeout}s (last status: {status})"
            )

        time.sleep(interval)


def fetch_run_download_url(run_id: str) -> str:
    """GET /api/runs/{id}/download-url and return the presigned download URL."""
    url = f"{_api_url()}/runs/{run_id}/download-url"
    log("remote_client", f"Fetching download URL for run {run_id} …")
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    download_url = res.json()["downloadUrl"]
    log("remote_client", f"Got download URL for run {run_id}")
    return download_url


def download_tarball(download_url: str, dest_dir: Path) -> Path:
    """Stream *download_url* into dest_dir/result.tar.gz and return the path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "result.tar.gz"

    log("remote_client", f"Downloading tarball …")
    with requests.get(download_url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(archive_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    log("remote_client", f"Tarball saved to {archive_path}")
    return archive_path


def extract_and_register_runrepo(archive_path: Path, dest_dir: Path) -> Path:
    """Extract *archive_path* into *dest_dir*, preserving the tarball's inner
    folder structure.  Sets globals.repo_dir to the inner run folder and enables
    PlayBack | HardPlayBack so the next run_experiment call replays the trace.

    Returns the inner run folder path.
    """
    from openfl.api import globals
    from openfl.api.globals import ReplayMode

    dest_dir.mkdir(parents=True, exist_ok=True)
    log("remote_client", f"Extracting tarball to {dest_dir} …")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)

    # Point repo_dir at the inner run folder (the folder inside the tarball)
    children = list(dest_dir.iterdir())
    run_dir = children[0] if len(children) == 1 and children[0].is_dir() else dest_dir

    log("remote_client", f"Run trace extracted to {run_dir}")

    globals.repo_dir = str(run_dir)
    globals.reuse_runs = ReplayMode.HardPlayBack | ReplayMode.PlayBack

    return run_dir


def run_remote_and_setup_replay(
    experiment_config,
    fingerprint: str,
    name: str | None = None,
    wanted_runs: int = 1,
    timeout: int = _POLL_TIMEOUT,
) -> Path:
    """Full remote pipeline: submit → poll → download → extract → register.

    Returns the extraction directory.  After this call, globals.repo_dir and
    globals.reuse_runs are configured so that experiment_runner.run_experiment
    will replay the remote result instead of training locally.
    """
    run_id = start_remote_experiment(experiment_config, fingerprint=fingerprint, name=name, wanted_runs=wanted_runs)
    poll_run_status(run_id, timeout=timeout)

    download_url = fetch_run_download_url(run_id)

    _experiment_dir = Path(__file__).resolve().parents[1]
    dest = _experiment_dir / "data" / "remote_runs" / fingerprint

    archive = download_tarball(download_url, dest)
    extract_dir = extract_and_register_runrepo(archive, dest)
    return extract_dir