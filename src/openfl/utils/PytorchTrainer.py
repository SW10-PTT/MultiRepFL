from __future__ import annotations

from typing import Tuple, OrderedDict, TYPE_CHECKING

import torch

# Imported only for type hints; skipped at runtime to avoid import errors when not on sys.path.
if TYPE_CHECKING:
    from web3.contract import Contract
from experiment.experiment_configuration import ExperimentConfiguration
from openfl.api.globals import ReplayMode, reuse_runs
from openfl.utils.ITestAndTrainer import ITestAndTrainer
import openfl.api.globals

class PyTorchTrainer(ITestAndTrainer):
    def __init__(self, config: ExperimentConfiguration, path="training_trace.json"):
        super().__init__(config, path)
    def train(self, round, tag, net, trainloader: torch.utils.data.DataLoader, epochs: int,
              device: torch.device) -> None:
        # nothing to save
        from openfl.ml.pytorch_model import train
        return train(net, trainloader, epochs, device)

    def test(self, round, tag, net, testloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[
            float, float]:
        from openfl.ml.pytorch_model import test
        data = test(net, testloader, device)
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, data)
        return data

    def get_hash(
            self,
            round: int,
            tag: str,
            model_state):
        from openfl.ml.pytorch_model import get_hash
        result = get_hash(model_state)
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, result)
        return result

    def on_chain_hashed_weights(self, round, tag, FLChallenge):
        result = {u.id: FLChallenge.get_hashed_weights_of(u) for u in FLChallenge.pytorch_model.participants}
        saveData = {str(key): value.hex() for (key, value) in result.items()}
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, saveData)
        return result

    def get_task_rep_delta_and_GRS(self, round, tag, contract: Contract, get_participant_func):
        data = contract.functions.getTaskRepDeltaAndGRS.call()
        formatted_data = [
            (str(get_participant_func(u[0]).id), u[1], u[2])
            for u in data
            if get_participant_func(u[0]) is not None
        ]
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, formatted_data)
        return data

    # NOT TO BE USED WITH MP
    def train_user_proc(
            self,
            round: int,
            tag: str,
            user_id, user_label, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle
    ):
        from openfl.ml.pytorch_model import train_user_proc
        result = train_user_proc(user_id, user_label, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory,
                                 shuffle)
        stripped_result = (result[0], result[2], result[3])
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, stripped_result)
        return result