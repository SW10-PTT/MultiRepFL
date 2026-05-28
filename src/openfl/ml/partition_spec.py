from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from openfl.utils.types.Attitude import Attitude


# Sentinel dataset key used when the JSON has a single-dataset (legacy) shape.
# Lookups fall back to this when no dataset-specific entry matches.
ANY_DATASET = "*"


# Behaviors allowed in per-user specs. Behavior is per-dataset, so a single
# user can be Honest on MNIST and Malicious on CIFAR-10. Inactive entries are
# permitted as well — they're skipped at user-creation time but counted toward
# the inactive contributor total.
_BEHAVIOR_MAP = {
    "honest": Attitude.Honest,
    "malicious": Attitude.Malicious,
    "freerider": Attitude.FreeRider,
    "inactive": Attitude.Inactive,
}
_BEHAVIOR_NAMES = "Honest, Malicious, Free-rider, Inactive"

# Behaviors that take noise_scale + start_round in the spec.
_NOISY_BEHAVIORS = (Attitude.Malicious, Attitude.FreeRider)


def parse_behavior(value) -> Attitude:
    if isinstance(value, Attitude):
        if value not in _BEHAVIOR_MAP.values():
            raise ValueError(
                f"behavior {value.name!r} not allowed in per-user specs; must be one of {_BEHAVIOR_NAMES}"
            )
        return value
    if not isinstance(value, str):
        raise TypeError(f"behavior must be a string, got {type(value).__name__}")
    norm = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm not in _BEHAVIOR_MAP:
        raise ValueError(
            f"behavior {value!r} invalid; must be one of {_BEHAVIOR_NAMES}"
        )
    return _BEHAVIOR_MAP[norm]


# Normalise dataset names so JSON keys ("MNIST", "CIFAR-10") match the runtime
# string set on experiment_config.dataset (e.g. "mnist", "cifar-10").
def normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return ANY_DATASET
    return str(name).strip().lower().replace(".", "-")


# Per-user partition spec for the per_user partition strategy. Immutable
# so it can flow through the config without surprise mutation.
# user_index is a free-form string identifier (GUID, numeric string, or any
# stable label). Any non-str input is coerced to str at construction time.
@dataclass(frozen=True)
class UserPartitionSpec:
    user_index: str
    data_percent: float
    label_distribution: Optional[Dict[int, float]] = None
    only_labels: Optional[List[int]] = None
    flip_map: Dict[int, int] = field(default_factory=dict)
    name: Optional[str] = None
    behavior: Attitude = Attitude.Honest
    # Required iff behavior in {Malicious, FreeRider}; forbidden otherwise.
    noise_scale: Optional[float] = None
    start_round: Optional[int] = None
    # Stable identifier propagated through the remote API so multirep can match
    # users across different Ganache instances.  Not part of fingerprint_dict().
    guid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        # Normalise to str so GUID and legacy-int keys land in the same shape.
        if not isinstance(self.user_index, str):
            object.__setattr__(self, "user_index", str(self.user_index))
        if not self.user_index:
            raise ValueError("user_index must be a non-empty string")
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

        # noise_scale + start_round are required for Malicious and Free-rider
        # entries (they drive per-user noise injection and the round at which
        # the user switches to its faulty behavior). Forbidden for Honest and
        # Inactive entries to avoid silent misconfiguration.
        if self.behavior in _NOISY_BEHAVIORS:
            if self.noise_scale is None:
                raise ValueError(
                    f"user {self.user_index}: behavior {self.behavior.name!r} requires "
                    f"'noise_scale' to be set in the spec"
                )
            if float(self.noise_scale) < 0:
                raise ValueError(
                    f"user {self.user_index}: noise_scale must be non-negative, got {self.noise_scale}"
                )
            if self.start_round is None:
                raise ValueError(
                    f"user {self.user_index}: behavior {self.behavior.name!r} requires "
                    f"'start_round' to be set in the spec"
                )
            if int(self.start_round) < 1:
                raise ValueError(
                    f"user {self.user_index}: start_round must be >= 1, got {self.start_round}"
                )
        else:
            if self.noise_scale is not None:
                raise ValueError(
                    f"user {self.user_index}: 'noise_scale' is not allowed when "
                    f"behavior is {self.behavior.name!r}"
                )
            if self.start_round is not None:
                raise ValueError(
                    f"user {self.user_index}: 'start_round' is not allowed when "
                    f"behavior is {self.behavior.name!r}"
                )

    def serialize(self) -> dict:
        """Full serialisation including guid — use when sending over the API."""
        return {**self.fingerprint_dict(), "guid": self.guid}

    # Deterministic dict for hashing in fingerprints. Keys/values are sorted
    # so the resulting JSON blob is byte-stable.  guid is intentionally excluded.
    def fingerprint_dict(self) -> dict:
        return {
            "user_index": str(self.user_index),
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
            "behavior": self.behavior.name,
            "noise_scale": (
                None if self.noise_scale is None else round(float(self.noise_scale), 8)
            ),
            "start_round": (
                None if self.start_round is None else int(self.start_round)
            ),
        }


# Single-dataset loader. Returns {user_index: UserPartitionSpec}. Accepts
# either a path to a JSON file or an in-memory dict in the legacy shape:
#   {"users": [{"user_index": "0", ...}, ...]}
#   {"<guid>": {...}, "<guid>": {...}}
def load_partition_specs(
    source: Union[str, Path, dict, None],
) -> Dict[str, UserPartitionSpec]:
    if source is None:
        return {}
    payload = _load_payload(source)
    return _load_single_dataset(payload)


# Multi-dataset loader. Returns {dataset_name_normalised: {user_index: spec}}.
# When the JSON has the legacy single-dataset shape, all specs land under
# ANY_DATASET so downstream lookups can fall back to it regardless of the
# active dataset. Canonical multi-dataset shape (array of users):
#   {"presets": [{"id": ..., "name": ..., "datasets": {DATASET: spec, ...}}, ...]}
# Legacy shapes still accepted:
#   {"users": [{"user_index": ..., "name": ..., "datasets": {...}}, ...]}
#   {user_index: {"name": ..., "datasets": {DATASET: spec, ...}}}
def load_dataset_partition_specs(
    source: Union[str, Path, dict, None],
) -> Dict[str, Dict[str, UserPartitionSpec]]:
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
    if "presets" in payload:
        entries = payload["presets"]
    elif "users" in payload:
        entries = payload["users"]
    else:
        entries = payload
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


def _load_multi_dataset(payload) -> Dict[str, Dict[str, UserPartitionSpec]]:
    if isinstance(payload, dict) and "presets" in payload:
        raw = payload["presets"]
    elif isinstance(payload, dict) and "users" in payload:
        raw = payload["users"]
    else:
        raw = payload

    out: Dict[str, Dict[str, UserPartitionSpec]] = {}

    def add_entry(user_key, entry):
        entry = dict(entry)
        if "user_index" not in entry and "id" not in entry and user_key is not None:
            entry["user_index"] = str(user_key)
        user_index = str(entry.get("user_index", entry.get("id")))
        name = entry.get("name")
        # Behavior, noise_scale and start_round are per-dataset by design.
        # Reject them at the user-entry level to keep the schema unambiguous.
        for forbidden in ("behavior", "noise_scale", "start_round"):
            if forbidden in entry:
                raise ValueError(
                    f"user {user_index}: {forbidden!r} must be set inside each dataset "
                    f"block, not on the user entry"
                )
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


def _load_single_dataset(payload) -> Dict[str, UserPartitionSpec]:
    if isinstance(payload, dict) and "presets" in payload:
        raw = payload["presets"]
    elif isinstance(payload, dict) and "users" in payload:
        raw = payload["users"]
    else:
        raw = payload

    specs: Dict[str, UserPartitionSpec] = {}
    if isinstance(raw, list):
        for entry in raw:
            spec = _build_spec(dict(entry))
            specs[spec.user_index] = spec
    elif isinstance(raw, dict):
        for key, entry in raw.items():
            entry = dict(entry)
            entry.setdefault("user_index", str(key))
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

    behavior_raw = entry.get("behavior")
    if behavior_raw is None:
        behavior = Attitude.Honest
    else:
        try:
            behavior = parse_behavior(behavior_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"user {entry['user_index']}: {e}"
            ) from None

    noise_scale = entry.get("noise_scale")
    if noise_scale is not None:
        noise_scale = float(noise_scale)

    start_round = entry.get("start_round")
    if start_round is not None:
        start_round = int(start_round)

    guid = entry.get("guid") or str(uuid.uuid4())

    return UserPartitionSpec(
        user_index=str(entry["user_index"]),
        data_percent=float(entry["data_percent"]),
        label_distribution=label_dist,
        only_labels=only_labels,
        flip_map=flip_map,
        name=name,
        behavior=behavior,
        noise_scale=noise_scale,
        start_round=start_round,
        guid=guid,
    )
