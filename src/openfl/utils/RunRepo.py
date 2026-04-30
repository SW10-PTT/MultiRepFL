import uuid
from typing import Tuple
from unittest import result

import torch
from hexbytes import HexBytes
from web3.contract import Contract

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
            user_addr, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle
    ):
        data = self.load(round, tag)
        expanded_data = (data[0], model_state, data[1], data[2])
        return expanded_data
    
    def get_task_rep_delta_and_GRS(self, round, tag, contract: Contract, get_participant_func):
        data = self.load(round, tag)
        formatted_data = [(get_participant_func(u[0]).address, u[1], u[2]) for u in data ]
        return formatted_data