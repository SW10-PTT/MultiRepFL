from openfl.api import ConnectionHelper, globals
from openfl.utils.types.TrainingSpecsJobListing import TrainingSpecsChallenge
from openfl.utils.types.User import User


class JobListing(ConnectionHelper):
    def __init__(self, publisher: User, training_specs: TrainingSpecsChallenge):

        w3 = globals.w3
        self.publisher = publisher

        # --- REQUIRED VALUES ---
        # modelHash = publisher.modelHash
        # assert modelHash is not None, "modelHash is missing"
        #
        # if not modelHash.startswith("0x"):
        #     modelHash = "0x" + modelHash
        #
        # model_hash_bytes = Web3.to_bytes(hexstr=modelHash)

        p1_collateral = publisher.collateral
        value = training_specs.reward + p1_collateral

        # --- FACTORY ---
        factory = self.initialize_job()

        # --- DEPLOY ---
        contract, receipt = ConnectionHelper.deploy(
            factory,
            [
                #model_hash_bytes,
                training_specs.min_collateral,
                training_specs.max_collateral,
                training_specs.reward,
                training_specs.min_rounds,
                training_specs.punishfactor,
                training_specs.punishfactorContrib,
                training_specs.freeriderPenalty,
                training_specs.manager_address,
                training_specs.taskType,
                training_specs.q_weight,
                training_specs.tr_weight,
                training_specs.gir_weight,
            ],
            publisher,
            value=value
        )

        self.contract = contract

    def register_challenge_contract(self, publisher, challenge_addr):
        (receipt, events) = self.transact("registerChallenge", publisher, 0, ["ChallengeRegistered"], "JobListing.RegisterChallengeContract",
                                          challenge_addr)

        is_valid = events["ChallengeRegistered"][0]["success"]

        return is_valid

    # Read-only fetch of the TaskType (= dataset) bound to this JobListing.
    def get_task_type(self) -> int:
        return self.contract.functions.getTaskType().call()
