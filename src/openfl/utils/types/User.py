import logging
import uuid

import numpy as np

from experiment.experiment_configuration import ExperimentConfiguration
from openfl.contracts import FLManager
from openfl.utils.async_writer import AsyncWriter
from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.Colors import RNG, get_color
from openfl.utils.types.TrainingSpecsJobListing import TrainingSpecsJobListing, TrainingSpecsChallenge
from openfl.api import globals
  
class User:
    user_count = 0

    def __init__(self,
                 _attitude, _default_collateral, _max_collateral,
                address, private_key, _attitude_switch=1, number_of_participants=None):
        if type(self) is User:
            self.number = User.user_count
            User.user_count += 1
        self.id = None
        self.address = address
        self.private_key = private_key
        # User's locally-trained model accuracy on their own validation set (after they trained on top of the global model).
        # Is set in: apply_training_results().
        # New variable introduced. Needs to be implemented in code. Alongside currentAcc.
        self.attitude = Attitude.Honest # Starts out honest
        self.futureAttitude = _attitude
        self.attitudeSwitch = _attitude_switch
        self.privateKey = None
        self.isRegistered = False
        self.min_collateral = _default_collateral
        self.max_collateral = _max_collateral
        self.txs = []
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

        self.color = get_color(number_of_participants, self.attitude)

    @classmethod
    def from_experiment_config(cls,
                               _attitude,
                               experiment_config: ExperimentConfiguration,
                               address, private_key,
                               number_of_participants=None):
        if _attitude == Attitude.Malicious:
            return User(_attitude, experiment_config.min_buy_in, experiment_config.max_buy_in,
                address, private_key, experiment_config.malicious_start_round, number_of_participants)
        return User(_attitude, experiment_config.min_buy_in, experiment_config.max_buy_in,
                address, private_key, experiment_config.freerider_start_round, number_of_participants)

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not callable(v) and not (k.startswith("_") or k.startswith("NOTHASH") or "loader" in k or "private" in k)
        }

    def get_status(self):
        user = f"$user${self.number}, {self.attitude}, {self.futureAttitude}, {self.attitudeSwitch}, {self.address}"
        return user

    def deploy_joblisting_contract(
        self,
        training_specs: TrainingSpecsJobListing,
        manager: "FLManager",
        ):
        from openfl.contracts.JobListing import JobListing
        
        new_job_listing = JobListing(self, training_specs)
        
        if manager.register_joblisting_contract(new_job_listing):
            return new_job_listing
        return False
    
    def deploy_challenge_contract(
        self,
        training_specs: TrainingSpecsChallenge,
        joblisting: "JobListing",
        pyTorch_model,
        writer: AsyncWriter = None,
        logger: logging.Logger = None,
        ):
        from openfl.contracts.FLChallenge import FLChallenge

        new_challenge = FLChallenge(self, pyTorch_model, training_specs, joblisting, writer, logger)
        
        if joblisting.register_challenge_contract(joblisting.publisher, new_challenge.contract.address):
            return new_challenge
        return False
    
    def register_for_job(self, job: "ConnectionHelper"):
        (receipt, _) = job.transact("register", self, self.collateral, [], "User.register_for_job")
        txHash = receipt["transactionHash"]
        self.txs.append(txHash)
        bal = globals.w3.eth.get_balance(globals.w3.eth.default_account)
        print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                self.address[0:16] + "...", 
                txHash.hex()[0:6] + "...", 
                self.collateral
                ))
        return self.txs

    def update_color(self, i, attitude):
        self.color = get_color(i, attitude)