from __future__ import annotations  # postpone annotation eval; allows forward refs without runtime imports

import copy
import uuid

import numpy as np
from web3 import Web3

from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.Colors import RNG, get_color
from openfl.utils.types.User import User


class Participant(User):
    def __init__(self, number, _train, _val, _model, _optimizer, _criterion, _attitude, _default_collateral,
                 _max_collateral, address, private_key, _data_percent, _only_labels, _attitude_switch=1, number_of_participants=None, participantId=None):
        super().__init__(_attitude, _default_collateral,
                         _max_collateral, address, private_key, _data_percent, _only_labels, _attitude_switch, number_of_participants)
        from openfl.api.ConnectionHelper import ConnectionHelper
        ConnectionHelper.initiate_connection(manual_setup=True)

        if isinstance(participantId, uuid.UUID):
            self.id = participantId
        elif isinstance(participantId, str):
            self.id = uuid.UUID(participantId)
        else:
            self.id = uuid.uuid4()

        self.number = number
        self.train = _train
        self.val = _val
        self.model = _model
        self.previousModel = copy.deepcopy(_model)
        self.modelHash = Web3.solidity_keccak(['string'], [str(_model)]).hex()
        self.optimizer = _optimizer
        self.criterion = _criterion
        self.userToEvaluate = []
        self.currentAcc = 0
        # User's locally-trained model accuracy on their own validation set (after they trained on top of the global model).
        # Is set in: apply_training_results().
        self.currentLoss = 0
        # New variable introduced. Needs to be implemented in code. Alongside currentAcc.
        self.attitude = Attitude.Honest
        self.hashedModel = None
        self.address = address
        self.isRegistered = False
        # Old:  self.collateral = _default_collateral + np.random.randint(0,int(_max_collateral-_default_collateral))
        # ---- collateral (handles huge ranges; avoids int32 cap) ----
        lo = int(_default_collateral)
        hi = int(_max_collateral)
        if hi < lo:
            raise ValueError(f"max_collateral ({hi}) must be >= default_collateral ({lo})")

        diff = hi - lo
        jitter = int(RNG.integers(0, np.int64(diff), dtype=np.int64)) if diff > 0 else 0
        self.collateral = lo + jitter

        # ---- secret (big nonce) ----
        self.secret = int(RNG.integers(0, np.int64(10 ** 18), dtype=np.int64))
        # self.secret = np.random.randint(0,int(1e18))

        self.roundRep = 0

        self.disqualified = False

        # INTERFACE VARIABLES - Not used for training. Only for reporting.
        self._accuracy = []  # User's accuracy on the global model. The actual accuracy evaluated on test set - is set in: finalize_user_evaluation().
        self._loss = []  # User's loss on the global model. The actual loss evaluated on test set - is set in: finalize_user_evaluation().
        self._globalrep = [self.collateral]
        self._roundrep = []
        self.txs = []

    def get_status(self):
        user = f"$user${self.number}, {str(self.id)}, {self.partition_name}, {self.currentAcc}, {self.attitude}, {self.futureAttitude}, {self.attitudeSwitch}, {self.address}"
        return user

    def from_user(user: User, train, val, model, optimizer, criterion):
        participant = Participant(
            user.number,
            train,
            val,
            model,
            optimizer,
            criterion,
            user.futureAttitude,
            user.min_collateral,
            user.max_collateral,
            user.address,
            user.private_key,
            user.data_percent,
            user.only_labels,
            user.attitudeSwitch,
            participantId=user.id)
        participant.partition_name = user.partition_name
        participant.partition_spec = user.partition_spec
        participant.noise_scale = user.noise_scale
        participant.start_round = user.start_round
        return participant

