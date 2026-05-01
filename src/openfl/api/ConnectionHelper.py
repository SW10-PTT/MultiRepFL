import json
import os
from pathlib import Path
import re
import os
import time
import signal
from web3 import Web3
from web3.contract import Contract
from termcolor import colored
from subprocess import Popen, PIPE

from openfl.ml.Participant import Participant
from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.Colors import gb, rb, b, green, red
from openfl.utils import require_env_var
from openfl.utils.printer import log
from openfl.api import globals

class ConnectionHelper:
    # Start Ganache client with connection to infura
    # Create web3 instance
    # Recursive function used to first get the latest block and then
    # ...fork the chain latest possible
    def initiate_rpc(self, 
                         NUMBER_OF_GOOD_CONTRIBUTORS, 
                         NUMBER_OF_BAD_CONTRIBUTORS, 
                         NUMBER_OF_FREERIDER_CONTRIBUTORS, 
                         NUMBER_OF_INACTIVE_CONTRIBUTORS,
                         MINIMUM_ROUNDS,
                         pytorch_model,
                         latestBlock=1000000, 
                         infura_url=None, 
                         manual_setup=False,
                         fork=True,
                         accounts=None):
        NUMBER_OF_CONTRIBUTORS = NUMBER_OF_GOOD_CONTRIBUTORS \
                                    + NUMBER_OF_BAD_CONTRIBUTORS \
                                    + NUMBER_OF_FREERIDER_CONTRIBUTORS \
                                    + NUMBER_OF_INACTIVE_CONTRIBUTORS

        
        
        #print("\n==================================================================================\n")
        log("setup_env", "Connected to Ethereum: {}".format(colored(globals.w3.is_connected(), "green", attrs=['bold'])))
        log("setup_env", "initiated Ganache-Client @ Block Nr. {:,.0f}\n".format(latestBlock))
        log("setup_contracts", "Total Contributers:       {}".format(NUMBER_OF_CONTRIBUTORS))
        log("setup_contracts", "Good Contributers:        {} ({:.0f}%)".format(NUMBER_OF_GOOD_CONTRIBUTORS,
                                                        NUMBER_OF_GOOD_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100))
        log("setup_contracts", "Malicious Contributers:   {} ({:.0f}%)".format(NUMBER_OF_BAD_CONTRIBUTORS,
                                                        NUMBER_OF_BAD_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100))
        log("setup_contracts", "Freeriding Contributers:  {} ({:.0f}%)".format(NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                        NUMBER_OF_FREERIDER_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100))
        log("setup_contracts", "Inactive Contributers:    {} ({:.0f}%)".format(NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                        NUMBER_OF_INACTIVE_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100))
        log("setup_contracts", "Learning Rounds:          {}".format(MINIMUM_ROUNDS))

        log("setup_contracts", "-----------------------------------------------------------------------------------")

        latestBlock = self.initiate_connection(NUMBER_OF_CONTRIBUTORS, latestBlock, manual_setup)

        if fork:
            while not globals.w3.eth.default_account:
                time.sleep(0.2)
                try:
                    globals.w3.eth.default_account = globals.w3.eth.accounts[0]
                except:
                    globals.w3.eth.default_account = None

            if len(globals.w3.eth.accounts) < len(self.pytorch_model.participants):
                print(rb("Nr. of Ganache Addresses <> Nr. of Model Participants"))
                print(rb(str(len(globals.w3.eth.accounts))  + "<>" +  str(len(self.pytorch_model.participants))))
                print(rb("Increase number of unlocked accounts"))
                raise NotEnoughUnlockedAccounts()

        # Every user receives an address
        for ix in range(len(self.pytorch_model.participants)):
            if globals.fork:
                self.pytorch_model.participants[ix].address = globals.w3.to_checksum_address(globals.w3.eth.accounts[ix])
            else:
                if ix == 0:
                    globals.w3.eth.default_account = accounts[ix].address
                self.pytorch_model.participants[ix].address = globals.w3.to_checksum_address(accounts[ix].address)
                self.pytorch_model.participants[ix].privateKey = accounts[ix].privateKey
                
            
        for i, acc in enumerate(self.pytorch_model.participants):
            if acc.futureAttitude == Attitude.Honest:
                prefix = "FAIR"
            elif acc.futureAttitude == Attitude.FreeRider:
                prefix = "FREE"
            elif acc.futureAttitude == Attitude.Inactive:
                prefix = "AFK "
            else:
                prefix = "MAL."
            bal = globals.w3.eth.get_balance(acc.address)
            log("setup_contracts", "{:<17} {} with {:<4,.1f} ETH | {} USER".format("Account initiated",
                                                           "@ Address "+acc.address[0:25]+"...",
                                                           bal/1e18,
                                                           prefix))
        log("setup_contracts", "-----------------------------------------------------------------------------------")
        return latestBlock

    @classmethod
    def initiate_connection(cls, latestBlock=1000000, manual_setup=False, NUMBER_OF_CONTRIBUTORS = 10):
        if globals.w3 is not None:
            return globals.w3.eth.block_number

        infura_url = require_env_var("RPC_URL")

        if globals.fork:
            if not manual_setup:
                port = require_env_var("RPC_URL").split(':')[1]
                process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
                stdout, stderr = process.communicate()
                for process in str(stdout.decode("utf-8")).split("\n")[1:]:
                    data = [x for x in process.split(" ") if x != '']
                    if len(data) <= 1:
                        continue

                    os.kill(int(data[1]), signal.SIGKILL)
                command = "ganache --fork.url='{}' -a {} -b 10".format(infura_url, NUMBER_OF_CONTRIBUTORS)
                os.system("gnome-terminal -e 'bash -c \"{}; bash\" '".format(command))
        while latestBlock == 1000000:
            time.sleep(1)
            try:
                if globals.fork:
                    globals.w3 = Web3(Web3.HTTPProvider(infura_url))
                    log("setup_env", "Connected:", globals.w3.is_connected())
                    log("setup_env", "Client:", globals.w3.client_version)
                    log("setup_env", "Chain ID:", globals.w3.eth.chain_id)
                    log("setup_env", "Latest block:", globals.w3.eth.block_number)
                    log("setup_env", "Accounts:", globals.w3.eth.accounts[:3])
                    log("setup_env", "Default account:", globals.w3.eth.default_account)
                    globals.w3.eth.default_account = globals.w3.eth.accounts[0]
                    log("setup_env", "New Default account:", globals.w3.eth.default_account)
                else:
                    globals.w3 = Web3(Web3.HTTPProvider(infura_url))
                latestBlock = globals.w3.eth.block_number
            except:
                latestBlock = 1000000

        return latestBlock;

    def initialize_manager(self):
        bytecode_path = Path(__file__).resolve().parents[3] / "artifacts" / "bytecode"
        with open(bytecode_path / "manager_abi.json") as abiFile:
            abi = json.load(abiFile)
        with open(bytecode_path / "manager_bytecode.bin") as bytecodeFile:
            bytecode = bytecodeFile.read().strip()
        return globals.w3.eth.contract(bytecode=bytecode, abi=abi)
    
    
    def initialize_challenge(self, address=None):
        bytecode_path = Path(__file__).resolve().parents[3] / "artifacts" / "bytecode"
        with open(bytecode_path / "model_abi.json") as abiFile:
            abi = re.sub("\n|\t| ", "", abiFile.read())
        with open(bytecode_path / "model_bytecode.bin") as abiFile:
            bytecode = abiFile.read().strip()
        if address is not None:
            return globals.w3.eth.contract(address=address, bytecode=bytecode, abi=abi)
        else:
            return globals.w3.eth.contract(bytecode=bytecode, abi=abi)
    
    def initialize_job(self, address=None) -> Contract:
        bytecode_path = Path(__file__).resolve().parents[3] / "artifacts" / "bytecode"
        with open(bytecode_path / "job_listing_abi.json") as abiFile:
            abi = re.sub("\n|\t| ", "", abiFile.read())
        with open(bytecode_path / "job_listing_bytecode.bin") as abiFile:
            bytecode = abiFile.read().strip()
        if address is not None:
            return globals.w3.eth.contract(address=address, bytecode=bytecode, abi=abi)
        else:
            return globals.w3.eth.contract(bytecode=bytecode, abi=abi)
    
    
    
    def build_tx(self, _from, _to, _value=0):
        assert(_to != "0x0000000000000000000000000000000000000000")
        _from = globals.w3.to_checksum_address(_from)
        _to = globals.w3.to_checksum_address(_to)
        return {
            'from': _from,
            'to': _to,
            'value': _value,
            #'gas': 300000,
            #'maxFeePerGas': self.w3.to_wei(250, 'gwei'),
            #'maxPriorityFeePerGas': self.w3.to_wei(5, 'gwei'),
        }
    
    
    
    def build_non_fork_tx(self, addr, nonce, to=None, value=0, data=None, gas_limit=None):
        # Dynamically detect correct chain ID
        chain_id = globals.w3.eth.chain_id

        # Give on-chain deployments breathing room unless caller overrides
        if gas_limit is None:
            gas_limit = 5_000_000

        # Adaptive low gas fee settings
        max_fee_per_gas = globals.w3.to_wei(10, 'gwei')
        max_priority_fee_per_gas = globals.w3.to_wei(1, 'gwei')

        # Check balance before building TX
        balance = globals.w3.eth.get_balance(addr)
        est_cost = gas_limit * max_fee_per_gas
        if balance < est_cost:
            print(f"\n Warning: Account {addr} has only {globals.w3.from_wei(balance, 'ether'):.4f} ETH, "
                f"but may need {globals.w3.from_wei(est_cost, 'ether'):.4f} ETH for gas.\n")

        # Build TX (same structure as before)
        if data:
            return {
                'chainId': chain_id,
                'from': addr,
                'to': to,
                'gas': gas_limit,
                'maxFeePerGas': max_fee_per_gas,
                'maxPriorityFeePerGas': max_priority_fee_per_gas,
                'nonce': nonce,
                'value': value,
                'data': data
            }

        if to:
            return {
                'chainId': chain_id,
                'from': addr,
                'to': to,
                'gas': gas_limit,
                'maxFeePerGas': max_fee_per_gas,
                'maxPriorityFeePerGas': max_priority_fee_per_gas,
                'nonce': nonce,
                'value': value
            }

        # Default case (no 'to' address)
        return {
            'chainId': chain_id,
            'from': addr,
            'gas': gas_limit,
            'maxFeePerGas': max_fee_per_gas,
            'maxPriorityFeePerGas': max_priority_fee_per_gas,
            'nonce': nonce,
            'value': value
        }

    def transact(self, func_name: str, account: Participant, collateral: int,  event_names: list[str], gas_type: str, *args):
        """
        Returns (receipt, event returns as tuple)
        funcname: contract function name
        acc: account performing the transaction
        event_names: names of the 
        *args: arguments forwarded to the contract function
        """
        return self.transact_raw_addreses(func_name, account.address, account.privateKey, collateral, event_names, gas_type, *args)

    def transact_raw_addreses(self, func_name: str, account_addr: str, account_private_key: str, collateral: int, event_names: list[str], gas_type: str, *args):
        """
        Returns (receipt, event returns as tuple)
        funcname: contract function name
        acc: account performing the transaction
        event_names: names of the
        *args: arguments forwarded to the contract function
        """
        if not isinstance(gas_type, str):
            raise Exception(f"Gas type {gas_type} not supported")

        func = getattr(self.contract.functions, func_name)

        if globals.fork:
            tx = self.build_tx(account_addr, self.contract.address, collateral)
            txHash = func(*args).transact(tx)
        else:
            nonce = globals.w3.eth.get_transaction_count(account_addr)
            # When building the transaction via contract ABI we must not pre-set the `to` field.
            depl = super().build_non_fork_tx(account_addr, nonce, value=collateral)
            depl = func(*args).build_transaction(depl)
            signed = globals.w3.eth.account.sign_transaction(depl, private_key=account_private_key)
            txHash = globals.w3.eth.send_raw_transaction(signed.raw_transaction)

        receipt = globals.w3.eth.wait_for_transaction_receipt(txHash, timeout=600, poll_latency=1)
        globals.add_gas_usage(gas_type, receipt["gasUsed"], account_addr)
        if receipt.get("status", 0) != 1:
            raise RuntimeError(
                f"Transaction: \"{func_name}\" failed (tx={txHash.hex()}, status={receipt.get('status')}). "
            )


        return (receipt, self.get_events(receipt, event_names))

        
    def get_events(self, receipt, event_names: list[str]):
        """
        Returns decoded events without ABI mismatch warnings.

        Args:
            receipt: transaction receipt
            event_names: list of event names to extract

        Returns:
            dict: {eventName: [decodedEvents...]}
        """
        results = {name: [] for name in event_names}

        for name in event_names:
            event_abi = getattr(self.contract.events, name)().abi
            event_signature = globals.w3.keccak(
                text=f"{name}(" + ",".join(i["type"] for i in event_abi["inputs"]) + ")").hex()

            for log in receipt.logs:
                if log["topics"][0].hex() == event_signature:
                    decoded = getattr(self.contract.events, name)().process_log(log)
                    results[name].append(decoded["args"])

        return results
    
    def deploy(factory, constructor_args, sender, value=0):
        w3 = globals.w3

        # --- FORK / LOCAL NODE (no private key needed) ---
        if globals.fork:
            tx_hash = factory.constructor(*constructor_args).transact({
                "from": sender.address,
                "value": value
            })

        # --- EXTERNAL SIGNING ---
        else:
            nonce = w3.eth.get_transaction_count(sender.address, "pending")

            tx = factory.constructor(*constructor_args).build_transaction({
                "from": sender.address,
                "nonce": nonce,
                "value": value,
                "gas": 3000000,
                "gasPrice": w3.eth.gas_price
            })

            signed = w3.eth.account.sign_transaction(
                tx,
                private_key=sender.privateKey
            )

            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        # --- RECEIPT ---
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.get("status", 0) != 1:
            raise RuntimeError(f"Deployment failed: {tx_hash.hex()}")

        address = Web3.to_checksum_address(receipt.contractAddress)

        contract = w3.eth.contract(address=address, abi=factory.abi)

        return contract, receipt
    
class NotEnoughUnlockedAccounts(Exception):
    pass