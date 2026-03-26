import copy

import numpy as np
from web3 import Web3
#from openfl.contracts import FLManager
from openfl.api.ConnectionHelper import ConnectionHelper
from openfl.utils.types.Colors import RNG, get_color
from openfl.utils.ChallengeTrainingSpecs import ChallengeTrainingSpecs
  
class Participant:
    def __init__(self, _id, _train, _val, _model, _optimizer, _criterion,
                 _attitude, _default_collateral, _max_collateral,
                 _attitudeSwitch=1, number_of_participants=None):
        self.id = _id
        self.train = _train
        self.val  = _val
        self.model = _model
        self.previousModel = copy.deepcopy(_model)
        self.modelHash = Web3.solidity_keccak(['string'],[str(_model)]).hex()
        self.optimizer = _optimizer
        self.criterion = _criterion
        self.userToEvaluate = []
        self.currentAcc = 0
        # User's locally-trained model accuracy on their own validation set (after they trained on top of the global model).
        # Is set in: apply_training_results().
        self.currentLoss = 0
        # New variable introduced. Needs to be implemented in code. Alongside currentAcc.
        self.attitude = "good"
        self.futureAttitude = _attitude
        self.attitudeSwitch = _attitudeSwitch
        self.hashedModel = None
        self.address = None
        self.privateKey = None
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

        self.color = get_color(number_of_participants, self.attitude)
        self.roundRep = 0

        self.disqualified = False

        # INTERFACE VARIABLES - Not used for training. Only for reporting.
        self._accuracy = [] # User's accuracy on the global model. The actual accuracy evaluated on test set - is set in: finalize_user_evaluation().
        self._loss = [] # User's loss on the global model. The actual loss evaluated on test set - is set in: finalize_user_evaluation().
        self._globalrep = [self.collateral]
        self._roundrep = []
        self.txs = []
    
    def getStatus(self):
        user = f"$user${self.id}, {self.currentAcc}, {self.attitude}, {self.futureAttitude}, {self.attitudeSwitch}, {self.address}"
        return user

    def deploy_joblisting_contract(
        self,
        trainingSpecs: ChallengeTrainingSpecs,
        manager
        ):
        from openfl.contracts.JobListing import JobListing
        
        newJobListing = JobListing(self, trainingSpecs)
        
        if manager.register_joblisting_contract(newJobListing):
            return newJobListing
        return False
    
    def register_for_job(self, job: ConnectionHelper):
        (receipt, _) = job.transact("register", self.address, self.collateral, [])
        txHash = receipt["transactionHash"]
        self.txs.append(txHash)
        bal = globals.w3.eth.get_balance(globals.w3.eth.default_account)
        print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                self.address[0:16] + "...", 
                txHash.hex()[0:6] + "...", 
                self.collateral
                ))
        return self.txs