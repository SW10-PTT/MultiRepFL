import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from experiment.multirep.MultirepRunConfig import MultirepRunConfig


@dataclass
class MultirepPreset:
    name: str
    partition_file: str
    tasks: List[MultirepRunConfig]

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MultirepPreset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = [MultirepRunConfig.from_dict(t) for t in data["tasks"]]
        return cls(
            name=data["name"],
            partition_file=data["partition_file"],
            tasks=tasks,
        )
