from __future__ import annotations

import datetime
import hashlib
import json
import random
import re
import socket
import uuid
from abc import ABC, abstractmethod
from dataclasses import is_dataclass, asdict
from pathlib import Path, PosixPath
from typing import Tuple, OrderedDict, List, TYPE_CHECKING

from hexbytes import HexBytes
from web3.contract import Contract
from openfl.api import globals

import torch

# Imported only for type hints; skipped at runtime to avoid import errors when not on sys.path.
if TYPE_CHECKING:
    from experiment.experiment_configuration import ExperimentConfiguration
from openfl.api.globals import ReplayMode
from openfl.utils.printer import log
from openfl.utils.types.Attitude import Attitude
from openfl.ml.Participant import Participant
from openfl.utils.types.User import User


class ITestAndTrainer(ABC):
    def __init__(self, config: ExperimentConfiguration, path="training_trace.json"):
        self._path = Path(path)

        self._data = {
            "participants": [],
            "config": config.to_dict(),
            "rounds": {}
        }

        if self._path.exists():
            with open(self._path, "r") as f:
                existing = json.load(f, object_hook=uuid_hook)

            existing = convert(existing)

            # Preserve old data but ensure config is present
            self._data["participants"] = existing.get("participants", [])
            self._data["rounds"] = existing.get("rounds", {})
            self._data["config"] = existing.get("config", config)

    @abstractmethod
    def train(
            self,
            round: int,
            tag: str,
            net: torch.nn.Module,
            trainloader: torch.utils.data.DataLoader,
            epochs: int,
            device: torch.device,
    ) -> None:
        """Train a model and record the result."""
        raise NotImplementedError


    @abstractmethod
    def train_user_proc(
            self,
            round: int,
            tag: str,
            user_addr, user_label, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle
    ):
        raise NotImplementedError

    @abstractmethod
    def get_hash(
            self,
            round: int,
            tag: str,
            model_state):
        raise NotImplementedError

    @abstractmethod
    def test(
            self,
            round: int,
            tag: str,
            net: torch.nn.Module,
            testloader: torch.utils.data.DataLoader,
            device: torch.device,
    ) -> Tuple[float, float]:
        """Evaluate a model and return (loss, accuracy)."""
        raise NotImplementedError

    def save(self, round: int, tag: str, obj):
        round = str(round)

        rounds = self._data["rounds"]

        if round not in rounds:
            rounds[round] = {}

        rounds[round][tag] = obj

    def flush(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=_serialize)

    def load(self, round: int = None, tag: str = None):
        if round is None:
            return self._data

        round = str(round)
        rounds = self._data.get("rounds", {})

        if tag is None:
            return rounds.get(round)

        return rounds.get(round, {}).get(tag)

    def set_participants(self, participants: List[Participant]):
        if self._data["participants"] == []:
            self._data["participants"] = [{"finger_print": p.finger_print, "id": p.id, "guid": p.guid} for p in participants]

    def get_participants(self, users: List[User]):
        usersRaw = self._data["participants"]
        used_addresses = set()
        used_users = set()

        for userRaw in usersRaw:
            userRawId = (
                userRaw["id"]
                if isinstance(userRaw["id"], uuid.UUID)
                else uuid.UUID(userRaw["id"])
            )

            for user in users:
                if (
                        user not in used_users
                        and userRawId not in used_addresses
                        and match_replay_user_user(userRaw, user)
                ):
                    used_addresses.add(userRawId)
                    used_users.add(user)
                    user.id = userRawId
                    break

        return [user for user in users if user.id is not None]
    
    @abstractmethod
    def on_chain_hashed_weights(self, round, tag, FLChallenge):
        pass

    @abstractmethod
    def get_task_rep_delta_and_GRS(self, round, tag, contract: Contract, get_participant_func):
        pass


def match_replay_user_user(replay_user, user: User):
    saved = replay_user.get("guid")
    return saved is not None and user.guid is not None and str(saved) == user.guid

def uuid_hook(obj):
    for k, v in obj.items():
        if isinstance(v, str):
            try:
                obj[k] = uuid.UUID(v)
            except ValueError:
                pass
    return obj

def convert(value):
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert(v) for v in value]
    elif isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError:
            return value
    elif isinstance(value, str) and value.startswith("0x"):
        try:
            return HexBytes(value)
        except Exception:
            return value
    return value

def _serialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Attitude):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, HexBytes):
        return "0x" + obj.hex()
    if isinstance(obj, PosixPath):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def _extract_fp_from_name(name: str) -> str | None:
    m = re.search(r'([0-9a-f]{64})\.json$', name)
    return m.group(1) if m else None


_FP_SCALAR_FIELDS = [
    "dataset", "minimum_rounds", "min_buy_in", "max_buy_in", "standard_buy_in",
    "epochs", "batch_size", "punish_factor", "punish_factor_contrib", "first_round_fee",
    "contribution_score_strategy", "loss_tolerance_pct", "use_outlier_detection",
    "freerider_noise_scale", "freerider_start_round", "malicious_start_round",
    "malicious_noise_scale", "force_merge_all", "seed", "allow_overlap",
    "replication_factor", "user_seeds", "partition_strategy", "vote_baseline",
    "global_rep_only", "tr_weight", "gir_weight",
]


def _load_fp_data_from_trace(trace_path: Path) -> dict | None:
    """Reconstruct the fingerprint data dict from a saved trace file."""
    try:
        with open(trace_path) as f:
            d = json.load(f)
        cfg = d.get("config", {})
        participants = sorted(p["finger_print"] for p in d.get("participants", []) if "finger_print" in p)
        result = {k: cfg.get(k) for k in _FP_SCALAR_FIELDS}
        result["participants"] = participants
        result["per_user_partitions"] = cfg.get("per_user_partitions")
        return result
    except Exception as e:
        log("replay", f"[fingerprint diff] could not load trace file {trace_path.name}: {e}")
        return None


def _resolve_participant_label(fp: str) -> str:
    label = globals.fp_user_labels.get(fp)
    return f"{label} ({fp[:8]}...)" if label else f"{fp[:8]}..."


def _log_fp_diff(local_fp: str, file_fp: str, local_data: dict, file_data: dict) -> None:
    lines = [f"[fingerprint diff] local={local_fp[:8]}... vs file={file_fp[:8]}..."]
    all_keys = sorted(set(local_data) | set(file_data))
    found_diff = False
    for k in all_keys:
        va, vb = local_data.get(k), file_data.get(k)
        if va == vb:
            continue
        found_diff = True
        if k == "participants":
            sa = set(va) if isinstance(va, list) else set()
            sb = set(vb) if isinstance(vb, list) else set()
            only_local = sa - sb
            only_file  = sb - sa
            if only_local:
                lines.append(f"  selected locally but NOT by chain: {[_resolve_participant_label(p) for p in sorted(only_local)]}")
            if only_file:
                lines.append(f"  selected by chain but NOT locally:  {[_resolve_participant_label(p) for p in sorted(only_file)]}")
            if only_local or only_file:
                lines.append(f"  → possible causes: tied scores (Python sort vs Solidity heap scan order) or rep-slot mismatch (Python and contract using different TaskType values)")
        elif k == "per_user_partitions":
            local_keys = sorted(va) if isinstance(va, dict) else []
            file_keys  = sorted(vb) if isinstance(vb, dict) else []
            lines.append(f"  per_user_partitions keys: local={local_keys}  file={file_keys}" if local_keys != file_keys
                         else f"  per_user_partitions: keys match but content differs (serialisation format may differ)")
        else:
            lines.append(f"  {k}: local={va!r}  file={vb!r}")
    if not found_diff:
        lines.append("  (no field-level differences found — per_user_partitions content may still differ)")
    log("replay", "\n".join(lines))


def get_filename(finger_print, config):
    if ReplayMode.PlayBack in globals.reuse_runs:
        search_dir = Path(globals.repo_dir)
        all_json = [f for f in search_dir.rglob("*.json") if f.is_file()]
        files = [f for f in all_json if f.name.endswith(f"{finger_print}.json")]
        random_file = random.choice(files) if files else None
        if random_file:
            globals.reuse_runs = globals.reuse_runs | ReplayMode._actively_replaying
            return random_file

        log("replay", f"[fallback] PlayBack set but no file matching fingerprint '{finger_print}' found in '{globals.repo_dir}'. "
                      f"Falling back to local training. "
                      f"Files present ({len(all_json)}): {[f.name for f in all_json[:10]]}{'...' if len(all_json) > 10 else ''}")

        local_data = globals.fp_data_cache.get(finger_print)
        for f in all_json[:3]:
            file_fp = _extract_fp_from_name(f.name)
            if not file_fp or file_fp == finger_print:
                continue
            file_data = globals.fp_data_cache.get(file_fp) or _load_fp_data_from_trace(f)
            if local_data and file_data:
                _log_fp_diff(finger_print, file_fp, local_data, file_data)
            elif not local_data:
                log("replay", f"[fingerprint diff] local fp {finger_print[:8]}... not in cache — fingerprint computed before cache was available")
            else:
                log("replay", f"[fingerprint diff] file fp {file_fp[:8]}... not loadable for comparison")

    return Path(globals.repo_dir) / make_filename(config, finger_print)


def make_filename(config: ExperimentConfiguration, finger_print) -> str:
    dateTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = uuid.uuid4().hex[:6]
    node_id = socket.gethostname()[:6]
    if config.name is not None:
        name = _to_pascal_case(config.name)
        return f"{name}-{dateTime}-{rand}-{node_id}-{finger_print}.json"
    else:
        return f"{dateTime}-{rand}-{node_id}-{finger_print}.json"

def _to_pascal_case(name: str, max_len=100) -> str:
    # Split on -, _, or whitespace
    parts = re.split(r"[-_\s]+", name.strip())

    # Capitalize each part and join
    pascal = "".join(p.capitalize() for p in parts if p)

    # Truncate for safety
    return pascal[:max_len]

def _hash_config(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
    