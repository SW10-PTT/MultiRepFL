from __future__ import annotations

from typing import Tuple, OrderedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from experiment.experiment_configuration import ExperimentConfiguration
from openfl.utils.ITestAndTrainer import ITestAndTrainer
from openfl.utils.types.ReplayTrainingSpecs import ReplayTrainingSpecs


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
        self.save(round, tag, data)
        return data

    def get_hash(
            self,
            round: int,
            tag: str,
            model_state):
        from openfl.ml.pytorch_model import get_hash
        result = get_hash(model_state)
        self.save(round, tag, result)
        return result

    def on_chain_hashed_weights(self, round, tag, FLChallenge):
        resultSave = {
            str(u.id): "0x" + FLChallenge.get_hashed_weights_of(u).hex()
            for u in FLChallenge.pytorch_model.participants
        }
        result = {u.id: FLChallenge.get_hashed_weights_of(u) for u in FLChallenge.pytorch_model.participants}
        self.save(round, tag, resultSave)
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
        self.save(round, tag, stripped_result)
        return result