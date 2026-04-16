from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from openfl.utils.types.AddressIndexList import AddressIndexList
from openfl.utils.types.AddressIndexMatrix import AddressIndexMatrix


class EvaluationData:
    def __init__(self,
        address_to_idx: Dict[str, int],
        feedback_matrix: AddressIndexMatrix,
        accuracy_matrix: AddressIndexMatrix,
        loss_matrix: AddressIndexMatrix,
        prev_accuracies: AddressIndexList,
        prev_losses: AddressIndexList,
    ):
        self.address_to_idx = address_to_idx
        self.idx_to_address = {idx: address for idx, address in enumerate(address_to_idx)}
        self.feedback_matrix = feedback_matrix
        self.accuracy_matrix = accuracy_matrix
        self.loss_matrix = loss_matrix
        self.prev_accuracies = prev_accuracies
        self.prev_losses = prev_losses

    @classmethod
    def new(cls, participants: List):
        address_to_idx = {
            p.address: i
            for i, p in enumerate(participants)
        }

        feedback_matrix = AddressIndexMatrix(external_address_list=address_to_idx)
        accuracy_matrix = AddressIndexMatrix(external_address_list=address_to_idx)
        loss_matrix = AddressIndexMatrix(external_address_list=address_to_idx)
        prev_accs = AddressIndexList(external_address_list=address_to_idx)
        prev_losses = AddressIndexList(external_address_list=address_to_idx)

        return cls(
            address_to_idx=address_to_idx,
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

    def get(self, address: str) -> "EvaluationData.UserVotes":
        idx = self.address_to_idx[address]


        return self.UserVotes(
            feedback={
                self.idx_to_address[i]: int(self.feedback_matrix[idx][i])
                for i in range(len(self.idx_to_address))
            },
            accuracy={
                self.idx_to_address[i]: int(self.accuracy_matrix[idx][i])
                for i in range(len(self.idx_to_address))
            } if self.accuracy_matrix is not None else None,
            loss={
                self.idx_to_address[i]: int(self.loss_matrix[idx][i])
                for i in range(len(self.idx_to_address))
            } if self.loss_matrix is not None else None,
        )

    def get_user_address(self, index: int):
        return self.idx_to_address[index]