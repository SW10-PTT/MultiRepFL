"""Load a multirep session.pkl produced by MultirepLogger."""

import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MultirepSession:
    session_id: str
    preset_name: str
    session_timestamp: str
    preset: dict
    reputation_timeline: pd.DataFrame
    global_accuracy: pd.DataFrame       # per-round accuracy across all tasks
    tasks: list[dict]

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def datasets(self) -> list[str]:
        return [t["dataset"] for t in self.tasks]

    def get_task_run_data(self, task_index: int) -> dict | None:
        """Return the embedded RunData tables for a specific task, or None."""
        for t in self.tasks:
            if t["task_index"] == task_index:
                return t.get("run_data")
        return None

    def iter_run_data(self):
        """Yield (task_index, dataset, tables_dict) for tasks that have embedded run data."""
        for t in self.tasks:
            rd = t.get("run_data")
            if rd is not None:
                yield t["task_index"], t["dataset"], rd


def _payload_to_session(payload: dict) -> MultirepSession:
    return MultirepSession(
        session_id=          payload["session_id"],
        preset_name=         payload["preset_name"],
        session_timestamp=   payload["session_timestamp"],
        preset=              payload.get("preset", {}),
        reputation_timeline= payload.get("reputation_timeline", pd.DataFrame()),
        global_accuracy=     payload.get("global_accuracy", pd.DataFrame()),
        tasks=               payload.get("tasks", []),
    )


def load_session(path: Path) -> MultirepSession:
    """Load a session from a session.pkl file or a session directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "session.pkl"
    with open(path, "rb") as f:
        return _payload_to_session(pickle.load(f))


def load_session_from_tarball(tarball: Path) -> MultirepSession:
    """Load a session directly from a .tar.gz archive without extracting to disk.

    Finds the first member whose name ends with 'session.pkl' and deserialises
    it from the in-archive byte stream.
    """
    tarball = Path(tarball)
    with tarfile.open(tarball, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith("session.pkl")),
            None,
        )
        if member is None:
            raise FileNotFoundError(f"session.pkl not found inside {tarball}")
        f = tar.extractfile(member)
        return _payload_to_session(pickle.load(f))
