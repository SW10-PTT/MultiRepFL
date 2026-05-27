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
            self._data["participants"] = [{"finger_print": p.finger_print, "id": p.id} for p in participants]

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
    return replay_user["finger_print"] == user.finger_print

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

def get_filename(finger_print, config):
    if ReplayMode.PlayBack in globals.reuse_runs:
        files = [f for f in list(Path(globals.repo_dir).glob("*.json")) if f.is_file() and f.name.endswith(f"{finger_print}.json")]
        random_file = random.choice(files) if files else None
        if random_file:
            globals.reuse_runs = globals.reuse_runs | ReplayMode._actively_replaying
            return random_file

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
    