import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from experiment.multirep.MultirepRunConfig import MultirepRunConfig
from experiment.multirep.training_mode import TrainingMode


@dataclass
class MultirepPreset:
    name: str
    partition_file: str
    tasks: List[MultirepRunConfig]

    # --- Selection scoring ---
    q_weight: float = 0.0
    tr_weight: int = 6
    gir_weight: int = 4
    # Cap on how many slots may be won via the Q bonus. Off by default; when off
    # selection is unchanged. When on, only q_slot_limit slots use the Q bonus —
    # the rest go to the highest base TR/GIR scores with no Q help.
    q_slot_limit_enabled: bool = False
    q_slot_limit: int = 0
    q_hard_reset: bool = False

    # --- Infrastructure ---
    training_mode: TrainingMode = TrainingMode.REMOTE
    fork: bool = True                       # True = Ganache fork, False = real net

    # --- Data partitioning (session-wide; override per-task values) ---
    replication_factor: float = 1.0         # data replication multiplier per user
    allow_overlap: bool = False             # whether user data partitions may overlap
    seed: int = 123                         # master RNG seed for data partitioning

    # --- Reputation system ---
    global_rep_only: bool = False           # True = one shared TR slot instead of per-task-type
    vote_baseline: str = "local_trained"    # reference for vote feedback; "local_trained" or "prev_global"

    # --- Remote scheduling ---
    priority: int | None = None             # worker claim priority; higher = claimed first
    force_remote: bool = True               # when True, retry remote forever instead of falling back to local

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MultirepPreset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = [MultirepRunConfig.from_dict(t) for t in data["tasks"]]
        raw_priority = data.get("priority")
        return cls(
            name=             data["name"],
            partition_file=   data["partition_file"],
            tasks=            tasks,
            q_weight=         float(data.get("q_weight", 0.0)),
            tr_weight=        int(data.get("tr_weight", 6)),
            gir_weight=       int(data.get("gir_weight", 4)),
            q_slot_limit_enabled= bool(data.get("q_slot_limit_enabled", False)),
            q_slot_limit=     int(data.get("q_slot_limit", 0)),
            q_hard_reset=     bool(data.get("q_hard_reset", False)),
            training_mode=    TrainingMode.from_string(data.get("training_mode", "remote")),
            fork=             bool(data.get("fork", True)),
            replication_factor= float(data.get("replication_factor", 1.0)),
            allow_overlap=    bool(data.get("allow_overlap", False)),
            seed=             int(data.get("seed", 123)),
            global_rep_only=  bool(data.get("global_rep_only", False)),
            vote_baseline=    str(data.get("vote_baseline", "local_trained")),
            priority=         int(raw_priority) if raw_priority is not None else None,
            force_remote=     bool(data.get("force_remote", True)),
        )
