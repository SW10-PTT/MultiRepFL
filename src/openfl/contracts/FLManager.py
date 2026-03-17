from eth_typing import ChecksumAddress
from web3 import Web3
from web3.contract import Contract
from openfl.contracts.JobListing import JobListing
from openfl.ml.pytorch_model import Participant, PytorchModel, b
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
        self.contract = super().initializeManager()
        return self
    
    
    # Deploy contract and initiate proxy
    def build_contract(self):
        manager_abi = self.contract.abi

        if globals.fork:
            genesisHash = self.contract.constructor().transact()  # Build Contract
        else:
            nonce = globals.w3.eth.get_transaction_count(globals.w3.eth.default_account) 
            depl = super().build_non_fork_tx(globals.w3.eth.default_account, nonce)   
            depl = self.contract.constructor().build_transaction(depl)
            signed = globals.w3.eth.account.sign_transaction(depl, private_key=self.pytorch_model.participants[0].privateKey)

            genesisHash = globals.w3.eth.send_raw_transaction(signed.raw_transaction)
            
        receipt = globals.w3.eth.wait_for_transaction_receipt(genesisHash,
                                                           timeout=600,
                                                           poll_latency=1)
        if receipt.get("status", 0) != 1:
            raise RuntimeError(
                f"Manager deployment failed (tx={genesisHash.hex()}, status={receipt.get('status')}). "
                "Check Sepolia gas settings and account balance."
            )

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildManager", receipt["transactionHash"].hex(), receipt["gasUsed"]))

        deployed_address = globals.w3.to_checksum_address(receipt.contractAddress)
        self.contract = globals.w3.eth.contract(address=deployed_address, abi=manager_abi)

        print("\n{:<17} {} | {}\n".format("Manager deployed", 
                                          "@ Address " + self.contract.address, 
                                          genesisHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        return 


    def get_model_of(self, participant, addr):
        return self.contract.functions.getModel(participant.address, addr).call({"to": self.contract.address,
                                                                  "from": participant.address})
    
    def deploy_joblisting_contract(self, publisher: Participant, *args) -> tuple[Contract, ChecksumAddress, JobListing, ...]:
        print(b("Creating Job Listing..."))
        print(b("-----------------------------------------------------------------------------------"))
        min_buyin, max_buyin, reward, min_rounds, punishment, punish_contrib, freerider_fee, taskType = args
        p1_collateral = publisher.collateral
        value = reward + p1_collateral
        publisher
        modelHash = self.pytorch_model.participants[0].modelHash
        model_hash_bytes = Web3.to_bytes(hexstr=modelHash)

        # Helpful debug info
        balance_eth = globals.w3.from_wei(globals.w3.eth.get_balance(publisher.address), 'ether')
        est_cost_eth = globals.w3.from_wei(value, 'ether')
        print(f"Balance: {balance_eth:.4f} ETH | Estimated tx+value cost: {est_cost_eth:.4f} ETH")

        (receipt, events) = self.transact("CreateNewJob", publisher, value, ["JobCreated"],
                model_hash_bytes,
                min_buyin,
                max_buyin,
                reward,
                min_rounds,
                punishment,
                punish_contrib,
                freerider_fee,
                taskType)
        
        txHash = receipt["transactionHash"]
        deployed_address = Web3.to_checksum_address(events["JobCreated"][0]["args"]["job"])
        newJobListing = JobListing(deployed_address)
        self.job_listings.append(newJobListing)
        
        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildJobListing", receipt["transactionHash"].hex(), receipt["gasUsed"]))
        
        
        print("\n{:<17} {} | {}\n".format("Listing deployed", 
                                          "@ Address " + newJobListing.contract.address, 
                                          txHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        print("{:<17} {} | {} | {:>25,.0f} WEI".format(
            "Account registered:",
            self.pytorch_model.participants[0].address[0:16] + "...", # No longer necessarily correct
            txHash.hex()[0:6] + "...",
            p1_collateral
        ))

        self.pytorch_model.participants[0].isRegistered = True # No longer necessarily correct
        return (newJobListing, (newJobListing.contract, newJobListing.contract.address) + args)