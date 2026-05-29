import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from experiment.multirep.MultirepRunConfig import MultirepRunConfig
from experiment.multirep.training_mode import TrainingMode


@dataclass
class MultirepPreset:
    name: str
    partition_file: str
    tasks: List[MultirepRunConfig]
    q_weight: float = 0.0
    tr_weight: int = 6
    gir_weight: int = 4
    training_mode: TrainingMode = TrainingMode.REMOTE

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MultirepPreset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = [MultirepRunConfig.from_dict(t) for t in data["tasks"]]
        return cls(
            name=data["name"],
            partition_file=data["partition_file"],
            tasks=tasks,
            q_weight=float(data.get("q_weight", 0.0)),
            tr_weight=int(data.get("tr_weight", 6)),
            gir_weight=int(data.get("gir_weight", 4)),
            training_mode=TrainingMode.from_string(data.get("training_mode", "remote")),
        )
