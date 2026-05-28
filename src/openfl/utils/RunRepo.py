from __future__ import annotations

import uuid
from typing import Tuple, TYPE_CHECKING
from unittest import result

import torch
from hexbytes import HexBytes
from web3.contract import Contract

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

    def flush(self):
        pass

    def train_user_proc(
            self,
            round: int,
            tag: str,
            user_addr, user_label, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle
    ):
        data = self.load(round, tag)
        expanded_data = (data[0], model_state, data[1], data[2])
        return expanded_data
    
    def get_task_rep_delta_and_GRS(self, round, tag, contract: Contract, get_participant_func):
        data = self.load(round, tag)
        # Recorded tuple shape is (user_id, delta, grs, positiveVotes, totalVotes).
        # Old traces written before GIR landed only carry the first three; pad
        # with zero vote tallies so replays of pre-GIR runs still work.
        formatted_data = [
            (get_participant_func(u[0]).guid, u[1], u[2])
            + (u[3] if len(u) > 3 else 0, u[4] if len(u) > 4 else 0)
            for u in data
        ]
        return formatted_data