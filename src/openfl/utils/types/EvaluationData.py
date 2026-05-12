from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from openfl.utils.types.AddressIndexList import AddressIndexList
from openfl.utils.types.AddressIndexMatrix import AddressIndexMatrix


class EvaluationData:
    def __init__(self,
                 id_to_idx: Dict[str, int],
                 feedback_matrix: AddressIndexMatrix,
                 accuracy_matrix: AddressIndexMatrix,
                 loss_matrix: AddressIndexMatrix,
                 prev_accuracies: AddressIndexList,
                 prev_losses: AddressIndexList,
                 ):
        self.id_to_idx = id_to_idx
        self.idx_to_id = {idx: id for idx, id in enumerate(id_to_idx)}
        self.feedback_matrix = feedback_matrix
        self.accuracy_matrix = accuracy_matrix
        self.loss_matrix = loss_matrix
        self.prev_accuracies = prev_accuracies
        self.prev_losses = prev_losses

    @classmethod
    def new(cls, participants: List):
        address_to_idx = {
            p.id: i
            for i, p in enumerate(participants)
        }
        id_to_label = {p.id: p.display_label() for p in participants if hasattr(p, "display_label")}

        feedback_matrix = AddressIndexMatrix(external_address_list=address_to_idx, np_int_type=np.int8, id_to_label=id_to_label)
        accuracy_matrix = AddressIndexMatrix(external_address_list=address_to_idx, id_to_label=id_to_label)
        loss_matrix = AddressIndexMatrix(external_address_list=address_to_idx, id_to_label=id_to_label)
        prev_accs = AddressIndexList(external_address_list=address_to_idx, id_to_label=id_to_label)
        prev_losses = AddressIndexList(external_address_list=address_to_idx, id_to_label=id_to_label)

        return cls(
            id_to_idx=address_to_idx,
            feedback_matrix=feedback_matrix,
            accuracy_matrix=accuracy_matrix,
            loss_matrix=loss_matrix,
            prev_accuracies=prev_accs,
            prev_losses=prev_losses,
        )

    @dataclass
    class UserVotes:
        feedback: Dict[str, int]
        accuracy: Optional[Dict[str, int]]
        loss: Optional[Dict[str, int]]
        prev_accuracy: Optional[int]
        prev_loss: Optional[int]

    def get(self, id: str) -> "EvaluationData.UserVotes":
        idx = self.id_to_idx[id]

        return self.UserVotes(
            feedback={
                self.idx_to_id[i]: int(self.feedback_matrix[idx][i])
                for i in range(len(self.idx_to_id))
            },
            accuracy={
                self.idx_to_id[i]: int(self.accuracy_matrix[idx][i])
                for i in range(len(self.idx_to_id))
            } if self.accuracy_matrix is not None else None,
            loss={
                self.idx_to_id[i]: int(self.loss_matrix[idx][i])
                for i in range(len(self.idx_to_id))
            } if self.loss_matrix is not None else None,
            prev_accuracy=int(self.prev_accuracies[idx]),
            prev_loss=int(self.prev_losses[idx]),
        )

    def get_user_id(self, index: int):
        return self.idx_to_id[index]