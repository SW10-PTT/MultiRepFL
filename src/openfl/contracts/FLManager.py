from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract
from openfl.contracts.JobListing import JobListing
from openfl.ml.pytorch_model import PytorchModel
from openfl.utils.types.Colors import b
from openfl.utils.types.Participant import Participant
from openfl.api import ConnectionHelper, globals

class FLManager(ConnectionHelper):
    def __init__(self, pytorch_model: PytorchModel, manual_ganache_setup=False):
        self.latestBlock = None
        self.contract = None
        self.challenge_contract = None
        self.pytorch_model: PytorchModel = pytorch_model
        self.modelOf = {}
        self.manual_setup = manual_ganache_setup
        self.job_listings = []

        self.gas_deploy = []
        self.txHashes   = []

        self.job_template_address = None
        #self.job_template_hash = None
        self.job_template_hash = Web3.to_bytes(hexstr="0xdb97405406fa6311775ff842c92fb4608768b2a54c37e98b4dad1adb090f27c2")

    def init(self, 
             NUMBER_OF_GOOD_CONTRIBUTORS, 
             NUMBER_OF_BAD_CONTRIBUTORS, 
             NUMBER_OF_FREERIDER_CONTRIBUTORS, NUMBER_OF_INACTIVE_CONTRIBUTORS, 
             MINIMUM_ROUNDS, 
             infuraurl=None, 
             accounts=None): 
        global w3
        self.latestBlock = super().initiate_rpc(NUMBER_OF_GOOD_CONTRIBUTORS=NUMBER_OF_GOOD_CONTRIBUTORS,
                                                         NUMBER_OF_BAD_CONTRIBUTORS=NUMBER_OF_BAD_CONTRIBUTORS,
                                                         NUMBER_OF_FREERIDER_CONTRIBUTORS=NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                         NUMBER_OF_INACTIVE_CONTRIBUTORS=NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                         MINIMUM_ROUNDS=MINIMUM_ROUNDS, pytorch_model=self.pytorch_model,
                                                         infura_url=infuraurl, manual_setup=self.manual_setup,
                                                         accounts=accounts)
        self.build_contract()

        self.deploy_job_template(self.pytorch_model.participants[0])
        
        self.transact("setJobListingCodeHash", self.pytorch_model.participants[0], 0, [], self.job_template_hash)
        return self
    
    
    # Deploy contract and initiate proxy
    def build_contract(self):
        factory = self.initialize_manager()

        contract, receipt = ConnectionHelper.deploy(
            factory,
            [],  # no constructor args
            self.pytorch_model.participants[0]
        )

        self.contract = contract

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildManager", receipt["transactionHash"].hex(), receipt["gasUsed"]))

        print("\n{:<17} {} | {}\n".format(
            "Manager deployed",
            "@ Address " + self.contract.address,
            receipt["transactionHash"].hex()[0:6] + "..."
        ))
        print("-----------------------------------------------------------------------------------")


    def get_model_of(self, participant, addr):
        return self.contract.functions.getModel(participant.address, addr).call({"to": self.contract.address,
                                                                  "from": participant.address})
    
    def register_joblisting_contract(self, new_joblisting: JobListing) -> bool:#-> tuple[Contract, ChecksumAddress, JobListing, ...]:
        (receipt, events) = self.transact("registerJob", new_joblisting.publisher, 0, ["JobListingValid"], new_joblisting.contract.address)
        
        is_valid = events["JobListingValid"][0]["args"]["isValid"]

        if not is_valid:
            return False
        
        self.job_listings.append(new_joblisting)
        return True
        # return (
        #     new_joblisting,
        #     (
        #         new_joblisting.contract,
        #         new_joblisting.contract.address,
        #         min_buyin,
        #         max_buyin,
        #         reward,
        #         min_rounds,
        #         punishment,
        #         punish_contrib,
        #         freerider_fee,
        #         taskType
        #     )
        # )
    
    def deploy_job_template(self, deployer: Participant):
        w3 = globals.w3

        factory = self.initialize_job()

        model_hash_bytes = Web3.keccak(text="template")  # any valid bytes32

        constructor_args = [
            model_hash_bytes,
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

        print("Job Listing template deployed at:", contract.address)
        print("Job Listing template hash:", self.job_template_hash.hex())

    def deploy_challenge_template(self, deployer: Participant):
        w3 = globals.w3

        factory = self.initialize_job()

        model_hash_bytes = Web3.keccak(text="template")  # any valid bytes32

        constructor_args = [
            model_hash_bytes,
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

        contract, receipt = self.deploy(
            factory,
            constructor_args,
            deployer,
            value=1
        )

        self.job_template_address = contract.address

        code = w3.eth.get_code(contract.address)
        self.job_template_hash = Web3.keccak(code)

        print("Job Listing template deployed at:", contract.address)
        print("Job Listing template hash:", self.job_template_hash.hex())