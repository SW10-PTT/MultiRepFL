"""Load a multirep session.pkl produced by MultirepLogger."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class MultirepSession:
    session_id: str
    preset_name: str
    session_timestamp: str
    preset: dict
    reputation_timeline: pd.DataFrame
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


def load_session(path: Path) -> MultirepSession:
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    return MultirepSession(
        session_id=        payload["session_id"],
        preset_name=       payload["preset_name"],
        session_timestamp= payload["session_timestamp"],
        preset=            payload.get("preset", {}),
        reputation_timeline= payload.get("reputation_timeline", pd.DataFrame()),
        tasks=             payload.get("tasks", []),
    )
