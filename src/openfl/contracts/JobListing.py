from typing import List

from web3 import Web3
from openfl.api import ConnectionHelper, globals
from openfl.ml.pytorch_model import PytorchModel
from openfl.utils.types.Colors import b
from openfl.utils.types.Participant import Participant
from openfl.utils import ChallengeTrainingSpecs


class JobListing(ConnectionHelper):
    def __init__(self, publisher: Participant, trainingSpecs: ChallengeTrainingSpecs):

        w3 = globals.w3
        self.publisher = publisher

        # --- REQUIRED VALUES ---
        modelHash = publisher.modelHash
        assert modelHash is not None, "modelHash is missing"

        if not modelHash.startswith("0x"):
            modelHash = "0x" + modelHash

        model_hash_bytes = Web3.to_bytes(hexstr=modelHash)

        p1_collateral = publisher.collateral
        value = trainingSpecs.reward + p1_collateral

        # --- FACTORY ---
        factory = self.initialize_job()

        # --- DEPLOY ---
        contract, receipt = ConnectionHelper.deploy(
            factory,
            [
                model_hash_bytes,
                trainingSpecs.min_collateral,
                trainingSpecs.max_collateral,
                trainingSpecs.reward,
                trainingSpecs.min_rounds,
                trainingSpecs.punishfactor,
                trainingSpecs.punishfactorContrib,
                trainingSpecs.freeriderPenalty,
                trainingSpecs.manager_address,
                trainingSpecs.taskType
            ],
            publisher,
            value=value
        )

        self.contract = contract


    def let_all_participants_register(self, participants: List):
        txs = []
        
        for acc in participants:
            if acc.isRegistered:
                continue

        (receipt, _) = self.transact("register", acc, acc.collateral, [])
        txHash = receipt["transactionHash"]
        txs.append(txHash)
        bal = globals.w3.eth.get_balance(globals.w3.eth.default_account)
        acc.isRegistered = True
        print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                acc.address[0:16] + "...", 
                txHash.hex()[0:6] + "...", 
                acc.collateral
                ))
        return txs
      