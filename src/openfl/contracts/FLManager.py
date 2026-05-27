from web3 import Web3
from openfl.contracts.JobListing import JobListing
from openfl.ml.pytorch_model import PytorchModel
from openfl.utils.types.TrainingSpecsJobListing import TrainingSpecsChallenge
from openfl.utils.types.User import User
from openfl.api import ConnectionHelper, globals
from openfl.utils.printer import log

class FLManager(ConnectionHelper):
    REPUTATION_MODE_PER_TASK = 0
    REPUTATION_MODE_GLOBAL_ONLY = 1

    def __init__(self, publisher, manual_ganache_setup=False, global_rep_only=False):

        self.latestBlock = None
        self.contract = None
        self.challenge_contract = None
        self.pytorch_model: PytorchModel
        self.modelOf = {}
        self.publisher = publisher
        self.manual_setup = manual_ganache_setup
        self.global_rep_only = bool(global_rep_only)
        self.job_listings = []

        self.gas_deploy = []
        self.txHashes   = []

        self.job_template_address = None
        #self.job_template_hash = None
        self.job_template_hash = Web3.to_bytes(hexstr="0xdb97405406fa6311775ff842c92fb4608768b2a54c37e98b4dad1adb090f27c2")
        self.challenge_templete_hash = Web3.to_bytes(hexstr="0xdb97405406fa6311775ff842c92fb4608768b2a54c37e98b4dad1adb090f27c2")

    def init(self,
             NUMBER_OF_GOOD_CONTRIBUTORS,
             NUMBER_OF_BAD_CONTRIBUTORS,
             NUMBER_OF_FREERIDER_CONTRIBUTORS, NUMBER_OF_INACTIVE_CONTRIBUTORS,
             MINIMUM_ROUNDS,
             infuraurl=None,
             accounts=None,
             existing_contract=None):
        # existing_contract: reuse an already-deployed OpenFLManager from a
        # previous run so on-chain reputation (TaskRep / GIR / task counters)
        # carries forward. When set, the manager contract + its job/challenge
        # template code hashes are left untouched (already configured) and only
        # this run's participant addresses are (re)assigned via initiate_rpc.
        self.latestBlock = super().initiate_rpc(NUMBER_OF_GOOD_CONTRIBUTORS=NUMBER_OF_GOOD_CONTRIBUTORS,
                                                         NUMBER_OF_BAD_CONTRIBUTORS=NUMBER_OF_BAD_CONTRIBUTORS,
                                                         NUMBER_OF_FREERIDER_CONTRIBUTORS=NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                         NUMBER_OF_INACTIVE_CONTRIBUTORS=NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                         MINIMUM_ROUNDS=MINIMUM_ROUNDS,
                                                         infura_url=infuraurl, manual_setup=self.manual_setup,
                                                         accounts=accounts)

        if existing_contract is not None:
            self.attach_existing(existing_contract)
            return self

        self.build_contract()

        self.deploy_job_template(self.publisher)
        self.deploy_challenge_template(self.publisher)

        self.transact("setJobListingCodeHash", self.publisher, 0, [], "Manager.Template.JobListing.SetHash", self.job_template_hash)
        self.transact("setChallengeCodeHash", self.publisher, 0, [], "JobListing.Template.Challenge.SetHash",self.challenge_templete_hash)
        return self

    # Bind this wrapper to a manager contract deployed by an earlier run
    # instead of deploying a fresh one. Guards that the reused contract's
    # immutable ReputationMode matches this run's global_rep_only setting —
    # mode can't change after deploy, so a mismatch means the caller is
    # threading the wrong manager through the run sequence.
    def attach_existing(self, manager_contract):
        on_chain_mode = manager_contract.functions.reputationMode().call()
        want_mode = (
            self.REPUTATION_MODE_GLOBAL_ONLY
            if self.global_rep_only
            else self.REPUTATION_MODE_PER_TASK
        )
        if on_chain_mode != want_mode:
            raise ValueError(
                "Reused OpenFLManager ReputationMode mismatch: "
                f"on-chain={on_chain_mode}, requested global_rep_only={self.global_rep_only} "
                f"(expected mode {want_mode}). A manager's mode is fixed at deploy time; "
                "do not mix PerTask and GlobalOnly runs on the same manager."
            )

        self.contract = manager_contract
        log("setup_contracts", "\n{:<17} {}\n".format(
            "Manager reused",
            "@ Address " + manager_contract.address
        ))
        log("setup_contracts", "-----------------------------------------------------------------------------------")
        return self
    
    
    # Deploy contract and initiate proxy
    def build_contract(self):
        factory = self.initialize_manager()

        reputation_mode = (
            self.REPUTATION_MODE_GLOBAL_ONLY
            if self.global_rep_only
            else self.REPUTATION_MODE_PER_TASK
        )

        contract, receipt = ConnectionHelper.deploy(
            factory,
            [reputation_mode],
            self.publisher
        )

        self.contract = contract

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildManager", receipt["transactionHash"].hex(), receipt["gasUsed"]))

        log("setup_contracts", "\n{:<17} {} | {}\n".format(
            "Manager deployed",
            "@ Address " + self.contract.address,
            receipt["transactionHash"].hex()[0:6] + "..."
        ))
        log("setup_contracts", "-----------------------------------------------------------------------------------")


    def get_model_of(self, participant, addr):
        return self.contract.functions.getModel(participant.address, addr).call({"to": self.contract.address,
                                                                  "from": participant.address})
    
    def register_joblisting_contract(self, new_joblisting: JobListing) -> bool:#-> tuple[Contract, ChecksumAddress, JobListing, ...]:
        (receipt, events) = self.transact("registerJob", new_joblisting.publisher, 0, ["JobListingValid"], "Manager.registerJob", new_joblisting.contract.address)

        is_valid = events["JobListingValid"][0]["isValid"]

        if not is_valid:
            return False
        
        self.job_listings.append(new_joblisting)
        return True
    
    def deploy_job_template(self, deployer: User):
        w3 = globals.w3

        factory = self.initialize_job()

        model_hash_bytes = Web3.keccak(text="template")  # any valid bytes32

        constructor_args = [
            #model_hash_bytes,
            1,   # min_buyin
            1,   # max_buyin
            1,   # reward
            1,   # min_rounds
            1,   # punishment
            1,   # punish_contrib
            1,   # freerider_fee
            self.contract.address if self.contract else deployer.address,  # manager addr
            0    # taskType (enum as int)
        ]

        contract, receipt = ConnectionHelper.deploy(
            factory,
            constructor_args,
            deployer,
            value=1
        )

        self.job_template_address = contract.address

        code = w3.eth.get_code(contract.address)
        self.job_template_hash = Web3.keccak(code)

        log("setup_contracts", "Job Listing template deployed at:", contract.address)
        log("setup_contracts", "Job Listing template hash:", self.job_template_hash.hex())

    def deploy_challenge_template(self, deployer: User):
        w3 = globals.w3

        factory = self.initialize_challenge()

        model_hash_bytes = Web3.keccak(text="template")  # any valid bytes32

        constructor_args = [
            TrainingSpecsChallenge(
                modelHash=model_hash_bytes,
                min_collateral=1,
                max_collateral=1,
                manager_address=deployer.address,
                reward=1,
                min_rounds=1,
                punishfactor=1,
                punishfactorContrib=1,
                freeriderPenalty=1,
                taskType=0,
                contribution_score_strategy=0,
                joblisting_address="0x0000000000000000000000000000000000000000",
                outlier_detection=False,
            ).to_solidity_challenge(),
        ]

        contract, receipt = ConnectionHelper.deploy(
            factory,
            constructor_args,
            deployer,
            value=1
        )

        code = w3.eth.get_code(contract.address)
        self.challenge_templete_hash = Web3.keccak(code)

        log("setup_contracts", "Challenge template deployed at:", contract.address)
        log("setup_contracts", "Challenge template hash:", self.challenge_templete_hash.hex())

    def update_reputations_from_challenge(self, challenge_address: str, task_type: int):
        """Sync reputation data from a completed challenge into the manager (Python/replay path).

        In production, the challenge calls the manager directly via finalizeReputations().
        In Python (replay or testing), call this after simulate() completes.
        """
        self.transact(
            "updateReputationsFromChallenge",
            self.publisher,
            0,
            [],
            "manager.updateReputationsFromChallenge",
            Web3.to_checksum_address(challenge_address),
            task_type,
        )
        log("setup_contracts", f"Manager reputation update applied from challenge {challenge_address[:10]}...")