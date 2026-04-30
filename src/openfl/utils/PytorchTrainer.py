from typing import Tuple

import torch

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
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, result)
        return result

    # NOT TO BE USED WITH MP
    def train_user_proc(
            self,
            round: int,
            tag: str,
            user_id, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle
    ):
        from openfl.ml.pytorch_model import train_user_proc
        result = train_user_proc(user_id, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory,
                                 shuffle)
        stripped_result = (result[0], result[2], result[3])
        if ReplayMode.Record in reuse_runs:
            self.save(round, tag, stripped_result)
        return result