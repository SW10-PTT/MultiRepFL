from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


# Per-user partition spec for the per_user partition strategy. Immutable
# so it can flow through the config without surprise mutation.
@dataclass(frozen=True)
class UserPartitionSpec:
    user_index: int
    data_percent: float
    label_distribution: Optional[Dict[int, float]] = None
    only_labels: Optional[List[int]] = None
    flip_map: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 < float(self.data_percent) <= 100.0):
            raise ValueError(
                f"user {self.user_index}: data_percent must be in (0, 100], got {self.data_percent}"
            )
        if self.label_distribution is not None:
            if not self.label_distribution:
                raise ValueError(
                    f"user {self.user_index}: label_distribution cannot be empty when provided"
                )
            if any(weight <= 0 for weight in self.label_distribution.values()):
                raise ValueError(
                    f"user {self.user_index}: label_distribution weights must be > 0"
                )
            if self.only_labels is not None:
                missing = set(self.label_distribution) - set(self.only_labels)
                if missing:
                    raise ValueError(
                        f"user {self.user_index}: label_distribution keys {sorted(missing)} not in only_labels"
                    )

    # Deterministic dict for hashing in fingerprints. Keys/values are sorted
    # so the resulting JSON blob is byte-stable.
    def fingerprint_dict(self) -> dict:
        return {
            "user_index": int(self.user_index),
            "data_percent": round(float(self.data_percent), 8),
            "label_distribution": (
                None
                if self.label_distribution is None
                else {
                    str(label): round(float(weight), 8)
                    for label, weight in sorted(self.label_distribution.items())
                }
            ),
            "only_labels": (
                None if self.only_labels is None else sorted(int(x) for x in self.only_labels)
            ),
            "flip_map": {str(src): int(dst) for src, dst in sorted(self.flip_map.items())},
        }


# Accepts either a path to a JSON file or an in-memory dict. Returns a
# {user_index: UserPartitionSpec} dict suitable for downstream lookups.
def load_partition_specs(
    source: Union[str, Path, dict, None],
) -> Dict[int, UserPartitionSpec]:
    if source is None:
        return {}

    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    elif isinstance(source, dict):
        payload = source
    else:
        raise TypeError(
            f"per_user_partitions must be str, Path, dict or None; got {type(source).__name__}"
        )

    # Two accepted JSON shapes:
    #   {"users": [{"user_index": 0, ...}, ...]}
    #   {"0": {...}, "1": {...}}  (also bare list/dict at root)
    if isinstance(payload, dict) and "users" in payload:
        raw = payload["users"]
    else:
        raw = payload

    specs: Dict[int, UserPartitionSpec] = {}
    if isinstance(raw, list):
        for entry in raw:
            spec = _build_spec(dict(entry))
            specs[spec.user_index] = spec
    elif isinstance(raw, dict):
        for key, entry in raw.items():
            entry = dict(entry)
            entry.setdefault("user_index", int(key))
            spec = _build_spec(entry)
            specs[spec.user_index] = spec
    else:
        raise ValueError(
            "per_user_partitions root must be a list of users or a {index: spec} mapping"
        )

    return specs


def _build_spec(entry: dict) -> UserPartitionSpec:
    # Accept "id" as an alias for "user_index" (matches the example JSON style).
    if "user_index" not in entry and "id" in entry:
        entry["user_index"] = entry["id"]
    if "user_index" not in entry:
        raise ValueError(f"partition entry missing 'user_index' (or 'id'): {entry}")

    label_dist = entry.get("label_distribution")
    if label_dist is not None:
        label_dist = {int(label): float(weight) for label, weight in label_dist.items()}

    only_labels = entry.get("only_labels")
    if only_labels is not None:
        only_labels = [int(x) for x in only_labels]

    flip_map = {int(src): int(dst) for src, dst in entry.get("flip_map", {}).items()}

    return UserPartitionSpec(
        user_index=int(entry["user_index"]),
        data_percent=float(entry["data_percent"]),
        label_distribution=label_dist,
        only_labels=only_labels,
        flip_map=flip_map,
    )
