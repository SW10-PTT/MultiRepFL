from __future__ import annotations

import uuid
from typing import Tuple, TYPE_CHECKING
from unittest import result

import torch
from hexbytes import HexBytes

# Imported only for type hints; skipped at runtime to avoid import errors when not on sys.path.
if TYPE_CHECKING:
    from experiment.experiment_configuration import ExperimentConfiguration
from openfl.utils.types.ReplayTrainingSpecs import ReplayTrainingSpecs
from openfl.utils.ITestAndTrainer import ITestAndTrainer


class RunRepo(ITestAndTrainer):
    def __init__(self, config: ExperimentConfiguration, path="training_trace.json"):
        super().__init__(config, path)

    def train(self, round, tag, net, trainloader: torch.utils.data.DataLoader, epochs: int,
              device: torch.device) -> None:
        # load - nothing to save
        pass

    def test(self, round, tag, net, testloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[
            float, float]:
        data = self.load(round, tag)
        return data

    def get_hash(
            self,
            round: int,
            tag: str,
            model_state):
        result = HexBytes(self.load(round, tag))
        return result

    def on_chain_hashed_weights(self, round, tag, FLChallenge):
        data = self.load(round, tag)
        return {
            uuid.UUID(k): HexBytes(v)
            for k, v in data.items()
        }

    def train_user_proc(
            self,
            round: int,
            tag: str,
            user_addr, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle
    ):
        data = self.load(round, tag)
        expanded_data = (data[0], model_state, data[1], data[2])
        return expanded_data