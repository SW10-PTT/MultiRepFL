from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


# Sentinel dataset key used when the JSON has a single-dataset (legacy) shape.
# Lookups fall back to this when no dataset-specific entry matches.
ANY_DATASET = "*"


# Normalise dataset names so JSON keys ("MNIST", "CIFAR-10") match the runtime
# string set on experiment_config.dataset (e.g. "mnist", "cifar-10").
def normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return ANY_DATASET
    return str(name).strip().lower().replace(".", "-")


# Per-user partition spec for the per_user partition strategy. Immutable
# so it can flow through the config without surprise mutation.
@dataclass(frozen=True)
class UserPartitionSpec:
    user_index: int
    data_percent: float
    label_distribution: Optional[Dict[int, float]] = None
    only_labels: Optional[List[int]] = None
    flip_map: Dict[int, int] = field(default_factory=dict)
    name: Optional[str] = None

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
            # Retention factors: 0 drops the class, 1 keeps the full fair-share
            # allocation, values in between subsample. Anything outside [0, 1]
            # is meaningless under the fair-share-then-filter semantics.
            if any(not 0.0 <= float(weight) <= 1.0 for weight in self.label_distribution.values()):
                raise ValueError(
                    f"user {self.user_index}: label_distribution weights must be in [0, 1]"
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
            "name": self.name,
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


# Single-dataset loader. Returns {user_index: UserPartitionSpec}. Accepts
# either a path to a JSON file or an in-memory dict in the legacy shape:
#   {"users": [{"user_index": 0, ...}, ...]}
#   {"0": {...}, "1": {...}}
def load_partition_specs(
    source: Union[str, Path, dict, None],
) -> Dict[int, UserPartitionSpec]:
    if source is None:
        return {}
    payload = _load_payload(source)
    return _load_single_dataset(payload)


# Multi-dataset loader. Returns {dataset_name_normalised: {user_index: spec}}.
# When the JSON has the legacy single-dataset shape, all specs land under
# ANY_DATASET so downstream lookups can fall back to it regardless of the
# active dataset. The new multi-dataset shape is:
#   {user_index: {"name": ..., "datasets": {DATASET: spec, ...}}}
#   {"users": [{"user_index": ..., "name": ..., "datasets": {...}}, ...]}
def load_dataset_partition_specs(
    source: Union[str, Path, dict, None],
) -> Dict[str, Dict[int, UserPartitionSpec]]:
    if source is None:
        return {}
    payload = _load_payload(source)
    if _is_multi_dataset(payload):
        return _load_multi_dataset(payload)
    return {ANY_DATASET: _load_single_dataset(payload)}


def _load_payload(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    elif isinstance(source, dict):
        payload = source
    else:
        raise TypeError(
            f"per_user_partitions must be str, Path, dict or None; got {type(source).__name__}"
        )

    # Strip documentation keys (anything starting with "_") at the root.
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    return payload


def _is_multi_dataset(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    entries = payload.get("users") if "users" in payload else payload
    if isinstance(entries, list):
        candidates = entries
    elif isinstance(entries, dict):
        candidates = entries.values()
    else:
        return False
    for entry in candidates:
        if isinstance(entry, dict) and "datasets" in entry:
            return True
    return False


def _load_multi_dataset(payload) -> Dict[str, Dict[int, UserPartitionSpec]]:
    raw = payload.get("users") if isinstance(payload, dict) and "users" in payload else payload

    out: Dict[str, Dict[int, UserPartitionSpec]] = {}

    def add_entry(user_key, entry):
        entry = dict(entry)
        if "user_index" not in entry and "id" not in entry and user_key is not None:
            entry["user_index"] = int(user_key)
        user_index = int(entry.get("user_index", entry.get("id")))
        name = entry.get("name")
        datasets = entry.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError(
                f"user {user_index}: multi-dataset entry must include a non-empty 'datasets' dict"
            )
        for dataset_name, spec_payload in datasets.items():
            spec_entry = dict(spec_payload)
            spec_entry.setdefault("user_index", user_index)
            spec_entry.setdefault("name", name)
            spec = _build_spec(spec_entry)
            key = normalize_dataset_name(dataset_name)
            out.setdefault(key, {})[spec.user_index] = spec

    if isinstance(raw, list):
        for entry in raw:
            add_entry(None, entry)
    elif isinstance(raw, dict):
        for key, entry in raw.items():
            add_entry(key, entry)
    else:
        raise ValueError(
            "per_user_partitions root must be a list of users or a {index: spec} mapping"
        )

    return out


def _load_single_dataset(payload) -> Dict[int, UserPartitionSpec]:
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

    name = entry.get("name")
    if name is not None:
        name = str(name)

    return UserPartitionSpec(
        user_index=int(entry["user_index"]),
        data_percent=float(entry["data_percent"]),
        label_distribution=label_dist,
        only_labels=only_labels,
        flip_map=flip_map,
        name=name,
    )
