import datetime
import os
import time
import warnings
from logging import Logger

import torch
from decimal import Decimal

from eth_abi import encode
from torch._numpy import uint16
from web3 import Web3
from web3.contract import Contract
from web3.types import TxReceipt
from termcolor import colored
import matplotlib.pyplot as plt
from web3.exceptions import ContractLogicError
from openfl.contracts import JobListing
from openfl.ml.pytorch_model import PytorchModel
from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.TrainingSpecsJobListing import TrainingSpecsChallenge
from openfl.utils.types.EvaluationData import EvaluationData
from openfl.utils.types.Colors import rb, b, green, red, yellow
from openfl.utils import printer, config
from openfl.utils.printer import log, fmt_floats, fmt_scaled_scores
from openfl.api.ConnectionHelper import ConnectionHelper
from openfl.api import globals
from openfl.utils.async_writer import AsyncWriter, NullWriter
from openfl.utils.shapley import check_shapley_compliance
from openfl.utils.types.User import User
from analysis import NullExperimentLogger

# Smart-contract–backed federated learning simulation.
# Handles:
#   - User registration / exit on-chain
#   - Hashed model submission & slot reservation
#   - Feedback exchange (reputation updates)
#   - Contribution score calculation (dot-product & MAD-based)
#   - Round settlement and visualization
UINT256_MAX = 2**256 - 1
UINT16_MAX = 2**16 - 1

from typing import List
import numpy as np




class FLChallenge(ConnectionHelper):
    def __init__(self, publisher: User, pyTorchModel, training_specs: TrainingSpecsChallenge, jobListing, writer: AsyncWriter=None, logger: Logger=None, manager_contract=None):

        self.pytorch_model: PytorchModel = pyTorchModel
        self.manager_contract = manager_contract
        self._publisher = publisher
        self.MIN_BUY_IN = training_specs.min_collateral
        self.MAX_BUY_IN = training_specs.max_collateral
        self.REWARD = training_specs.reward
        self.MIN_ROUNDS = training_specs.min_rounds
        self.PUNISHMENT_FACTOR = training_specs.punishfactor
        self.PUNISHMENT_FACTOR_CONTRIB = training_specs.punishfactorContrib
        self.FREERIDER_FACTOR = training_specs.freeriderPenalty
        
        self.contribution_score_strategy = training_specs.contribution_score_strategy
        self.loss_tolerance_pct = getattr(training_specs, "loss_tolerance_pct", 0.1)
        self.use_outlier_detection = training_specs.outlier_detection
        self.scores = []
        self.gas_feedback = []
        self.gas_register = []
        self.gas_slot     = []
        self.gas_weights  = []
        self.gas_close    = []
        self.gas_deploy   = []
        self.gas_exit     = []
        self.gas_contrib  = []
        self.txHashes     = []

        self._reward_balance = [self.REWARD]
        self._punishments = []
        self.config = config.get_contracts_config()
        self.writer = writer or NullWriter()
        self._logger = logger or NullExperimentLogger()
        self.writeTxProgress = 0


        self._contribution_score_strategy = training_specs.contribution_score_strategy
        self._contribution_score_calculators = {
            "dotproduct": self._calculate_scores_dotproduct,
            "naive": self._calculate_scores_naive,
            "accuracy_loss": self._calculate_scores_accuracy_loss,
            "accuracy_only": self._calculate_scores_accuracy_only,
            "loss_only": self._calculate_scores_loss_only,
            "loss_tolerance_aware": self._calculate_scores_loss_tolerance_aware,
            "loss_tolerance_snap": self._calculate_scores_loss_tolerance_snap,
        }

        self.disqualifiedUserEvents = []

        factory = self.initialize_challenge()

        p1_collateral = publisher.collateral
        value = training_specs.reward + p1_collateral

        # --- DEPLOY ---
        contract, receipt = ConnectionHelper.deploy(
            factory,
            [
                training_specs.to_solidity_challenge()
            ],
            publisher,
            value=value
        )


        self.contract: Contract = contract
        self.contractAddress = contract.address
        log("setup_contracts", "Contract address:", self.contract.address)
        log("setup_contracts", "Contract ABI functions:", [f["name"] for f in self.contract.abi if f["type"] == "function"])

        if training_specs.taskType == 0:
            log("setup_contracts", "Contract is template")
            return
        self.participant_addresses = jobListing.contract.functions.getSelectedParticipants.call()


    def _get_contribution_score_calculator(self):
        """
        Return the function used for contribution-score calculation,
        based on the configured strategy.
        """

        strategy = self._contribution_score_strategy
        if strategy not in self._contribution_score_calculators:
            available = ", ".join(sorted(self._contribution_score_calculators))
            raise ValueError(
                f"Unknown contribution score strategy '{strategy}'. Available strategies: {available}"
            )
        log("round_scoring", "strategy: ", strategy)
        return self._contribution_score_calculators[strategy]
        
    
    def get_hashed_weights_of(self, user):
        return self.contract.functions.weightsOf(user.address,self.pytorch_model.round-1).call({"to": self.contractAddress})
    
    def _label_for_address(self, addr):
        # Resolve a participant/disqualified user's display_label() from on-chain
        # event data (which only carries the address). Returns "?" when not found.
        for u in self.pytorch_model.participants + self.pytorch_model.disqualified:
            if u.address == addr:
                return u.display_label()
        return "?"

    def get_global_reputation_of_user(self, userAddr):
        user = self.contract.functions.users(userAddr).call()
        return user[1]
    
    def get_round_reputation_of_user(self, user):
        user_struct = self.contract.functions.users(user).call()
        return user_struct[3]

    def get_reward_left(self):
        return self.contract.functions.rewardLeft().call({"to": self.contractAddress})

    def users_provide_hashed_weights(self):
        txs = []
        label_w = max((len(acc.display_label()) for acc in self.pytorch_model.participants), default=12)
        for acc in self.pytorch_model.participants:
            if acc.attitude == "inactive":
                log("round_models", "{:<17}  {:<{lw}} ({}) | {} | {:>25,.0f} WEI".format("Account inactive:",
                                                                         acc.display_label(),
                                                                         acc.address[0:16] + "...",
                                                                         "  ...  ",
                                                                         self.get_global_reputation_of_user(acc.address),
                                                                         lw=label_w))
                continue
            if globals.fork:
                tx = super().build_tx(acc.address, self.contractAddress, 0)
                txHash = self.contract.functions.provideHashedWeights(acc.hashedModel, acc.secret).transact(tx) # Todo: use self.transact

            else:
                nonce = globals.w3.eth.get_transaction_count(acc.address)
                hw = super().build_non_fork_tx(acc.address, nonce)
                hw =  self.contract.functions.provideHashedWeights(acc.hashedModel, acc.secret).build_transaction(hw)
                signed = globals.w3.eth.account.sign_transaction(hw, private_key=acc.privateKey)
                txHash = globals.w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            log("round_models", "{:<17}  {:<{lw}} ({}) | {} | {:>25,.0f} WEI".format("Weights provided:",
                                                                         acc.display_label(),
                                                                         acc.address[0:16] + "...",
                                                                         txHash.hex()[0:6] + "...",
                                                                         self.get_global_reputation_of_user(acc.address),
                                                                         lw=label_w))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar("round_models", i, l)
            receipt = globals.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, # WTF IS THIS wait properly please
                                                            poll_latency=1)
            
            self.gas_weights.append(receipt["gasUsed"])
            self.txHashes.append(("weights", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            self._log_receipt(receipt, "weights")
        log("round_models", "-----------------------------------------------------------------------------------\n")
        

             
    def give_feedback(self, feedbackGiver, target, score):
        """
        Send a feedback transaction from feedbackGiver to target with given score:
          1  -> positive
          0  -> neutral
         -1  -> negative

        If target is in feedbackGiver.cheater list, force score to -1.
        """
        global fork
        global w3
        time.sleep(0.1)
        tx = super().build_tx(feedbackGiver.address, self.contractAddress, 0)
        #data = "0x" + encode_abi(['address', 'uint'], [target, score]).hex()
        if target in feedbackGiver.cheater:
            score = -1
        try:
            if fork:
                txHash = self.contract.functions.feedback(target.address, score).transact(tx) # Todo: use self.transact
            else:          
                nonce = w3.eth.get_transaction_count(feedbackGiver.address)
                fe = super().build_non_fork_tx(feedbackGiver.address, nonce)
                fe =  self.contract.functions.feedback(target.address, score).build_transaction(fe)
                signed = w3.eth.account.sign_transaction(fe, private_key=feedbackGiver.privateKey)
                txHash = w3.eth.send_raw_transaction(signed.raw_transaction)
        except ContractLogicError as e:
            if "FRC" in str(e):
                input("Inactive users found - such users do not provide hashed weights.. \nGoing to forward time for 1 day\n")
                w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
                time.sleep(1)
                txHash = self.contract.functions.feedback(target.address, score).transact(tx) # Todo: use self.transact
            else:
                print(rb("Encountered error at feedback function"))
                raise 
                
        assert(txHash != None)

        receipt = globals.w3.eth.wait_for_transaction_receipt(txHash, timeout=600, poll_latency=1)
        self.gas_feedback.append(receipt["gasUsed"])
        self.txHashes.append(("feedback", receipt["transactionHash"].hex(), receipt["gasUsed"]))
        self._log_receipt(receipt, "feedback")

        if score == 1:
            target.roundRep += 1 * self.get_global_reputation_of_user(feedbackGiver.address)
            rep = "Positive"
            pre = "+"
            col = "green"

        elif score == 0:
            rep = "Neutral"
            pre = "+"
            col = None
        else:
            target.roundRep -= 1 * self.get_global_reputation_of_user(feedbackGiver.address)
            rep = "Negative"
            pre = "-"
            col = "red"
        fb = "Feedback:".format(rep)
        
        log("round_models",colored("{:<11} {}   |" \
            " {}  | {}{:>25,.0f} WEI".format(fb,
                                    f"{feedbackGiver.display_label()} ({feedbackGiver.address[0:7]}...) --> {target.display_label()} ({target.address[0:7]}...)",
                                    txHash.hex()[0:6] + "...",
                                    pre,
                                    self.get_global_reputation_of_user(feedbackGiver.address)), col))
        return txHash
        
            
    
    def return_stats(self):
        log("round_rewards","\n==================================================================================\n")
        log("round_rewards","\n{:<24}{:^32}  {:^32}".format(f"ROUND {self.pytorch_model.round}","GLOBAL REPUTATION", "ROUND REPUTATION"))
        for acc in self.pytorch_model.participants:
            gs = self.get_global_reputation_of_user(acc.address)
            rs = self.get_round_reputation_of_user(acc.address)
            label = f"{acc.display_label()} ({acc.address[0:7]}..)"
            log("round_rewards","{:<24}: {:>27,.0f}  {:>27,.0f} WEI".format(label, gs, rs))
        log("round_rewards","\n==================================================================================\n")
    
            
    def feedback_round(self, fbm):
        txs = []
        for user in self.pytorch_model.participants:
            user_votes = fbm[user.id]
            for ix, vote in enumerate(user_votes):
                if user.id == ix:
                    continue
                if user.attitude == "inactive":
                    continue
                txHash = self.give_feedback(user, self.pytorch_model.participants[ix], int(vote))
                txs.append(txHash)
           
        l = len(txs)
        for i, txHash in enumerate(txs):
            if txHash == None:
                continue
            printer.print_bar("round_models", i, l)
            receipt = globals.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_feedback.append(receipt["gasUsed"])
            self.txHashes.append(("round_models", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            self._log_receipt(receipt, "round_models")
        for user in self.pytorch_model.participants:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))

        for user in self.pytorch_model.disqualified:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))
        log("round_models", "                                                   ")
        log("round_models", "\n-----------------------------------------------------------------------------------")

    def build_feedback_bytes(self, a, v):
        fbb = ""  # keep as string

        # Addresses: slice last 20 bytes to mimic original behavior
        for addr in a:
            encoded_addr = encode(["address"], [addr])  # 32 bytes
            fbb += encoded_addr.hex()[24:]  # take last 20 bytes in hex

        # Integers: full 32 bytes
        for val in v:
            fbb += encode(["int256"], [val]).hex()

        return fbb

    def quick_feedback_round(self, matrices: EvaluationData):
        log("round_models", "Users exchanging feedback...")
        txs = []

        disqualified_addr = {
            u.address
            for u in self.pytorch_model.disqualified
        }

        for user in self.pytorch_model.participants:

            user_id = user.id
            if user.disqualified:
                continue

            addrs = []
            votes = []
            filtered_accs = []
            filtered_losses = []

            user_votes = matrices.feedback_matrix[user_id]

            accs = matrices.accuracy_matrix[user_id]
            losses = matrices.loss_matrix[user_id]

            for idx, vote in enumerate(user_votes):
                addr = matrices.get_user_id(idx)
                if addr == user_id:
                    continue

                if addr in disqualified_addr:
                    continue

                if user.attitude == Attitude.Inactive:
                    continue

                votee = self.pytorch_model.get_participant(matrices.get_user_id(idx))

                if votee is None:
                    continue

                addrs.append(votee.address)
                votes.append(int(vote))

                rep_delta = self.get_global_reputation_of_user(user.address) * int(vote)
                votee.roundRep += rep_delta
                votee._roundrep.append(rep_delta)

                if accs is not None:
                    filtered_accs.append(matrices.accuracy_matrix[user_id, matrices.get_user_id(idx)].item())

                if losses is not None:
                    filtered_losses.append(min(UINT256_MAX, matrices.loss_matrix[user_id, matrices.get_user_id(idx)].item()))

            fbb = self.build_feedback_bytes(addrs, votes)
            rb_fbb = Web3.to_bytes(hexstr="0x" + fbb)

            if self.contribution_score_strategy in ["naive", "dotproduct"]:
                if globals.fork:
                    tx = super().build_tx(user.address, self.contractAddress)
                    tx_hash = self.contract.functions.submitFeedbackBytes(
                        rb_fbb
                    ).transact(tx) # Todo: use self.transact
                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.contract.functions.submitFeedbackBytes(rb_fbb)
                    )
                txs.append(tx_hash)

            elif self.contribution_score_strategy == "accuracy_loss":
                if globals.fork:
                    tx = super().build_tx(user.address, self.contractAddress)
                    tx_hash = self.contract.functions.submitFeedbackBytesAndAccuraciesLosses(
                        rb_fbb, filtered_accs, filtered_losses, matrices.prev_accuracies[user_id].tolist(), matrices.prev_losses[user_id].tolist()
                    ).transact(tx) # Todo: use self.transact
                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.contract.functions.submitFeedbackBytesAndAccuraciesLosses(
                            rb_fbb, filtered_accs, filtered_losses, matrices.prev_accuracies[user_id].tolist(), matrices.prev_losses[user_id].tolist()
                        )
                    )
                txs.append(tx_hash)

            elif self.contribution_score_strategy == "accuracy_only":
                prev_acc = matrices.prev_accuracies[user_id]

                if globals.fork:
                    tx = super().build_tx(user.address, self.contractAddress)
                    tx_hash = self.contract.functions.submitFeedbackBytesAndAccuracies(
                        rb_fbb, filtered_accs, prev_acc
                    ).transact(tx) # Todo: use self.transact
                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.contract.functions.submitFeedbackBytesAndAccuracies(
                            rb_fbb, filtered_accs, prev_acc
                        )
                    )
                txs.append(tx_hash)

            elif self.contribution_score_strategy in ("loss_only", "loss_tolerance_aware", "loss_tolerance_snap"):
                prev_loss = int(min(matrices.prev_losses[user_id], 10000))  # contract requires [0, 10000]

                if globals.fork:
                    tx = super().build_tx(user.address, self.contractAddress)
                    tx_hash = self.contract.functions.submitFeedbackBytesAndLosses(
                        rb_fbb, filtered_losses, prev_loss
                    ).transact(tx) # Todo: use self.transact
                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.contract.functions.submitFeedbackBytesAndLosses(
                            rb_fbb, filtered_losses, prev_loss
                        )
                    )
                txs.append(tx_hash)

            else:
                warnings.warn("INVALID FEEDBACK TYPE")

        l = len(txs)
        for i, tx_hash in enumerate(txs):
            if tx_hash is None:
                continue
            printer.print_bar("round_models", i, l)
            receipt = globals.w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=600, poll_latency=1
            )
            self.gas_feedback.append(receipt["gasUsed"])
            self.txHashes.append(("feedback", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            self._log_receipt(receipt, "feedback")

        # Append aggregated on-chain roundReputation so _roundrep[-1] reflects the
        # post-voting total (matches analysis/README "round reputation after voting").
        # Per-vote deltas above remain in the list for any historical inspection.
        for user in self.pytorch_model.participants + self.pytorch_model.disqualified:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))


    def sign_and_send_tx(self, user, contract_fn_call):
        nonce = globals.w3.eth.get_transaction_count(user.address)
        tx = super().build_non_fork_tx(user.address, nonce)
        tx = contract_fn_call.build_transaction(tx)

        signed = globals.w3.eth.account.sign_transaction(tx, private_key=user.privateKey)
        return globals.w3.eth.send_raw_transaction(signed.raw_transaction)


    def track_transaction(self, i, tx_hash, len_txs, receipt_type: str):  # formerly named log_receipt
        #   1. Prints a progress bar — i out of len_txs transactions done
        #   2. Waits for the transaction to be mined — blocks until the receipt comes back (up to 600s timeout)
        #   3. Stores gas used — appends to self.gas_contrib (sole caller is submitContributionScore)
        #   4. Stores the tx hash + gas — appends to self.txHashes along with the receipt_type label (e.g. "contrib")

        printer.print_bar("round_scoring", i, len_txs)
        receipt = globals.w3.eth.wait_for_transaction_receipt(tx_hash,
                                                           timeout=600,
                                                           poll_latency=1)

        self.gas_contrib.append(receipt["gasUsed"])
        self.txHashes.append((receipt_type, receipt["transactionHash"].hex(), receipt["gasUsed"]))
        # Writer (old logger) uses this to log

        self._log_receipt(receipt, receipt_type)
        # New logger log this way


    def send_fallback_transaction_onchain(self, _to, _from, data, private_key=None):
        try:
            if globals.fork:
                tx_hash = globals.w3.eth.send_transaction({'to': _to, 'from': _from, 'data': data})
            else:
                nonce = globals.w3.eth.get_transaction_count(_from)
                hw = super().build_non_fork_tx(_from, nonce, self.contractAddress, 0, data)
                signed = globals.w3.eth.account.sign_transaction(hw, private_key=private_key)
                tx_hash = globals.w3.eth.send_raw_transaction(signed.raw_transaction)

        except ContractLogicError as e:
            if "FRC" in str(e):
                input("Inactive users found - such users do not " \
                      + "provide hashed weights.. \nGoing to forward time for 1 day\n")

                globals.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
                time.sleep(1)
                tx_hash = globals.w3.eth.send_transaction({'to': _to,
                                                       'from': _from,
                                                       'data': data,
                                                       "gas": 500000})
            else:
                print(rb("Encountered error at feedback function"))
                raise
        return tx_hash

    def close_round(self):
        if "inactive" in [acc.attitude for acc in self.pytorch_model.participants]:
                input("Inactive users found - such users do not provide feedback.. " \
                          + "\nGoing to forward time for 1 day\n")
                globals.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
        
        log("round_boundary", b(f"Feedback round: {self.pytorch_model.round}"))
        settleStart = datetime.datetime.now(datetime.timezone.utc).timestamp()
        while (datetime.datetime.now(datetime.timezone.utc).timestamp() < settleStart + config.get_contracts_config().FEEDBACK_ROUND_TIMEOUT):
            if (self.contract.functions.isFeedBackRoundDone().call()):
                log("round_boundary", "Feedback round completed")
                break
            log("round_boundary", "Feedback round not done, sleeping for 10 seconds...")
            time.sleep(10)
        else:
            log("round_boundary", "Feedback round failed, forcing Contribution...")

        log("round_boundary", b(f"Contribution round: {self.pytorch_model.round}"))
        contributionStart = datetime.datetime.now(datetime.timezone.utc).timestamp()
        while (datetime.datetime.now(datetime.timezone.utc).timestamp() < contributionStart + config.get_contracts_config().CONTRIBUTION_ROUND_TIMEOUT):
            if (self.contract.functions.isContributionRoundDone().call()):
                log("round_boundary", "Contribution round completed")
                break
            log("round_boundary", "Contribution round not done, sleeping for 10 seconds...")
            time.sleep(10)
        else:
            log("round_boundary", "Contribution round failed, forcing settlement...")


        log("round_boundary", b(f"Settling round: {self.pytorch_model.round}"))
        if globals.fork:
            tx = super().build_tx(globals.w3.eth.default_account, self.contractAddress, 0)
            txHash = self.contract.functions.settle().transact(tx)
        else:
            nonce = globals.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address, 'pending')
            cl = super().build_non_fork_tx(self.pytorch_model.participants[0].address, nonce)
            cl =  self.contract.functions.settle().build_transaction(cl)
            pk = self.pytorch_model.participants[0].privateKey
            signed = globals.w3.eth.account.sign_transaction(cl, private_key=pk)
            txHash = globals.w3.eth.send_raw_transaction(signed.raw_transaction)

        receipt = globals.w3.eth.wait_for_transaction_receipt(txHash,
                                                        timeout=600,
                                                        poll_latency=1)
        log("round_boundary", "settling round completed")

        self.txHashes.append(("close", receipt["transactionHash"].hex(), receipt["gasUsed"]))
        self.gas_close.append(receipt["gasUsed"])
        self._log_receipt(receipt, "close")
        if len(receipt.logs) == 0:
            log("round_boundary", "Warning: closeFeedBackRound() emitted no logs")
        self.pytorch_model.round += 1
        self._reward_balance.append(self.get_reward_left())
        log("round_boundary", "\n-----------------------------------------------------------------------------------\n")
        return receipt

    def users_register_slot(self):
        txs = []
        label_w = max((len(acc.display_label()) for acc in self.pytorch_model.participants), default=12)
        for acc in self.pytorch_model.participants:
            if acc.attitude == "inactive":
                log("round_models","{:<17}  {:<{lw}} ({}) | {} | {:>25,.0f} WEI".format("Account inactive:",
                                                                         acc.display_label(),
                                                                         acc.address[0:16] + "...",
                                                                         "  ...  ",
                                                                         self.get_global_reputation_of_user(acc.address),
                                                                         lw=label_w))
                continue

            reservation = Web3.solidity_keccak(['bytes32', 'uint256', 'address'],
                                              [acc.hashedModel,
                                               acc.secret, acc.address])
            if globals.fork:
                tx = super().build_tx(acc.address, self.contractAddress, 0)
                txHash = self.contract.functions.registerSlot(reservation).transact(tx) # Todo: use self.transact
            else:
                w3 = ConnectionHelper.get_w3()          
                nonce = w3.eth.get_transaction_count(acc.address) 
                sl = super().build_non_fork_tx(acc.address, nonce)
                sl =  self.contract.functions.registerSlot(reservation).build_transaction(sl)
                signed = w3.eth.account.sign_transaction(sl, private_key=acc.privateKey)
                txHash = w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            log("round_models", "{:<17}  {:<{lw}} ({}) | {} | {:>25,.0f} WEI".format("Slot registered: ",
                                                                         acc.display_label(),
                                                                         acc.address[0:16] + "...",
                                                                         txHash.hex()[0:6] + "...",
                                                                         self.get_global_reputation_of_user(acc.address),
                                                                         lw=label_w))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar("round_models", i, l)
            receipt = globals.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600,
                                                            poll_latency=1)
            
            self.gas_slot.append(receipt["gasUsed"])
            self.txHashes.append(("slot", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            self._log_receipt(receipt, "slot")
        log("round_models", "-----------------------------------------------------------------------------------\n")
        return
    
    
    
    def compute_and_record_task_reps(self, caller):
        """Calculate and store per-participant TaskRep updates on-chain.

        Guarded: requires at least one round settled (round > 0) and idempotent.
        Called once per challenge run; applies updates to the manager in one call.
        """
        (receipt, events) = self.transact(
            "computeAndRecordTaskReps",
            caller,
            0,
            [],
            "challenge.computeAndRecordTaskReps",
        )
        return receipt, events

    def get_task_rep_records(self):
        """Read the stored TaskRepRecord[] from the challenge (after computeAndRecordTaskReps)."""
        return self.contract.functions.getTaskRepRecords().call()

    def _finalize_reputations(self):
        """Compute, store, and apply TR updates for this challenge.

        Two-step: challenge computes+stores records; publisher key applies them
        to the manager directly (avoids bytecode-hash auth on the challenge side).
        """
        if self.manager_contract is None:
            return
        try:
            self.compute_and_record_task_reps(self._publisher)
        except Exception as e:
            log("setup_contracts", f"[warn] computeAndRecordTaskReps failed: {e}")
            return
        try:
            records = self.get_task_rep_records()
            task_type = self.contract.functions.taskType().call()
            # manager_contract is the FLManager wrapper, not a raw web3 contract.
            # Route the write through its transact() so it targets the manager's
            # web3 contract (self.contract) and handles both fork (unlocked
            # accounts) and non-fork (signed) modes. Sent from the publisher, which
            # is the manager's deployer, so it passes the on-chain auth check
            # (msg.sender == publisher).
            self.manager_contract.transact(
                "applyPrecomputedTaskReps",
                self._publisher,
                0,
                [],
                "manager.applyPrecomputedTaskReps",
                records,
                task_type,
            )
            log("setup_contracts", f"TR applied to manager: {len(records)} records task_type={task_type}")
        except Exception as e:
            log("setup_contracts", f"[warn] applyPrecomputedTaskReps failed: {e}")

    def exit_system(self):

        log("experiment_end", b(f"Terminating Model..."))

        txs = []
        label_w = max((len(acc.display_label()) for acc in self.pytorch_model.participants), default=12)
        for acc in self.pytorch_model.participants:

            if globals.fork:
                tx = super().build_tx(acc.address, self.contractAddress, 0)
                txHash = self.contract.functions.exitModel().transact(tx) # Todo: use self.transact
            else:
                w3 = ConnectionHelper.get_w3()
                nonce = w3.eth.get_transaction_count(acc.address)
                ex = super().build_non_fork_tx(acc.address, nonce)
                ex =  self.contract.functions.exitModel().build_transaction(ex)
                signed = w3.eth.account.sign_transaction(ex, private_key=acc.privateKey)
                txHash = w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            log("experiment_end", "{:<17}  {:<{lw}} ({}) | {} | {:>32,.0f} WEI".format("Account exited:  ",
                                                             acc.display_label(),
                                                             acc.address[0:16] + "...",
                                                             txHash.hex()[0:6] + "...",
                                                             globals.w3.eth.get_balance(acc.address),
                                                             lw=label_w))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar("experiment_end", i, l)
            receipt = globals.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_exit.append(receipt["gasUsed"])
            self.txHashes.append(("exit", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            self._log_receipt(receipt, "exit")
        log("experiment_end", "-----------------------------------------------------------------------------------\n")

    def print_round_summary(self, receipt):

        events = self.get_events(
            receipt=receipt,
            event_names=["EndRound", "Reward", "Punishment", "Disqualification"]
        )

        end_events = events["EndRound"]
        reward_events = events["Reward"]
        punish_events = events["Punishment"]
        disqualify_events = events["Disqualification"]

        # End of round summary
        if end_events:
            for ev in end_events:
                log("round_boundary", b(f"\nEND OF ROUND {ev['round'] + 1}"))
                log("round_boundary", b(f"VALID VOTES:      {ev['validVotes']}"))
                log("round_boundary", b(f"SUM OF WEIGHTS:  {ev['sumOfWeightedContribScore']:,}"))
                log("round_boundary", b(f"TOTAL PUNISHMENT: {ev['totalPunishment']:,}\n"))
            log("round_boundary", "-----------------------------------------------------------------------------------\n")

        # Rewarded users
        if reward_events:
            log("round_rewards", b("REWARDED USERS"))
            for ev in reward_events:
                if ev["roundScore"] > 0:
                        log("round_rewards", green(f"USER @ {self._label_for_address(ev['user'])} ({ev['user']})"))
                        log("round_rewards", green(f"ROUND SCORE:      {ev['roundScore']:,}"))
                        total_reward = ev['win']
                        if not ev.get('is_reward', True):  # default True if key missing
                            total_reward = -total_reward
                        log("round_rewards", green(f"TOTAL REWARD:     {ev['win']:,}"))
                        log("round_rewards", green(f"NEW REPUTATION:   {ev['newReputation']:,}\n"))
            log("round_rewards", "-----------------------------------------------------------------------------------\n")

        # Punished users
        if punish_events:
            log("round_rewards", b("PUNISHED USERS"))
            for ev in punish_events:
                log("round_rewards", "Punishing a user")
                self._punishments.append((
                    self.pytorch_model.round - 1,
                    ev["loss"],
                    next((i + 1 for i, x in enumerate(self.pytorch_model.participants) if x.address == ev["victim"]), 0),
                    ))
                log("round_rewards", red(f"USER @ {self._label_for_address(ev['victim'])} ({ev['victim']})"))
                log("round_rewards", red(f"ROUND SCORE:      {ev['roundScore']:,}"))
                log("round_rewards", red(f"TOTAL LOSS:       {ev['loss']:,}"))
                log("round_rewards", red(f"NEW REPUTATION:   {ev['newReputation']:,}\n"))
            log("round_rewards", "-----------------------------------------------------------------------------------\n")

        # Disqualified users
        if disqualify_events:
            log("round_rewards", b("DISQUALIFIED USERS"))
            for ev in disqualify_events:
                log("round_rewards", "Disqualifying a user")
                self._punishments.append((
                    self.pytorch_model.round - 1,
                    ev["loss"],
                    next((i + 1 for i, x in enumerate(self.pytorch_model.participants) if x.address == ev["victim"]), 0)),
                    )

                # Mark and remove disqualified users
                for user in list(self.pytorch_model.participants):  # safe remove
                    if ev["victim"] == user.address:
                        user.disqualified = True
                        self.pytorch_model.disqualified.append(user)
                        self.pytorch_model.participants.remove(user)

                log("round_rewards", red(f"USER @ {self._label_for_address(ev['victim'])} ({ev['victim']})"))
                log("round_rewards", red(f"ROUND SCORE:      {ev['roundScore']:,}"))
                log("round_rewards", red(f"TOTAL LOSS:       {ev['loss']:,}"))
                log("round_rewards", red(f"NEW REPUTATION:   {ev['newReputation']:,}\n"))
            log("round_rewards", "-----------------------------------------------------------------------------------\n")

        log("round_boundary")


    def contribution_score(self, _users):
        """
        Compute contribution scores for all merging users, submit them to the
        contract, and log them. Strategy is chosen by _get_contribution_score_calculator:
          - legacy: simple dot-product
          - mad: MAD-based outlier filtering of weights
          - naive: equal-share (1 / num_mergers)
        """

        # Guard: no users → nothing to score
        if not _users:
            log("round_scoring", "-----------------------------------------------------------------------------------")
            log("round_scoring", "No users passed to contribution_score – skipping.")
            log("round_scoring", "-----------------------------------------------------------------------------------")
            return

        log("round_scoring", "START CONTRIBUTION SCORE\n")

        # Choose scoring algorithm based on configured strategy
        calculator = self._get_contribution_score_calculator()
        self.scores = calculator(_users)

        txs = []
        for u, score in zip(_users, self.scores):
            u.is_contrib_score_negative = True if score < 0 else False
            u.contribution_score = score

            if globals.fork:
                tx = super().build_tx(u.address, self.contractAddress)
                tx_hash = self.contract.functions.submitContributionScore(
                    score
                ).transact(tx) # Todo: use self.transact
            else:  # TODO: Dobbeltjek at logic er rigtig her.
                nonce = globals.w3.eth.get_transaction_count(u.address)
                cl = super().build_non_fork_tx(
                    u.address,
                    nonce,
                )
                cl = self.contract.functions.submitContributionScore(
                    score,
                ).build_transaction(cl)
                pk = u.privateKey
                signed = globals.w3.eth.account.sign_transaction(cl, private_key=pk)
                tx_hash = globals.w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(tx_hash)

            log("round_scoring", green(f"\nUSER @ {u.display_label()} (#{u.number})"))
            if u. is_contrib_score_negative:
                log("round_scoring", red(f"{'NEGATIVE CONTRIBUTION SCORE:':25}{u.contribution_score}"))
            else:
                log("round_scoring", green(f"{'CONTRIBUTION SCORE:':25}{u.contribution_score}"))

        for i, txHash in enumerate(txs):
            self.track_transaction(i, txHash, len(txs), "contrib")

        log("round_scoring", "-----------------------------------------------------------------------------------\n")


    def _calculate_scores_dotproduct(self, users):
        """
        MAD-based scoring: robust per-weight outlier filtering before scoring.
        """
        merged_model = users[0].model
        global_update = torch.cat([p.data.view(-1) for p in merged_model.parameters()])
        local_updates = [
            torch.cat([p.data.view(-1) for p in u.previousModel.parameters()]) for u in users
        ]
        local_updates = torch.stack(local_updates)


        if self.use_outlier_detection:
            log("round_scoring", "using mad")
            filtered_global_update, per_user_outlier_info = self.trim_global_update_using_mad(local_updates, global_update)
            scores = calc_contribution_scores_dotproduct(local_updates, filtered_global_update)

            # Raw dot product per user (pre-normalization), analogous to avg_acc/avg_loss in other strategies
            dots = torch.mv(local_updates, filtered_global_update)
            raw_values = [float(d.item()) for d in dots]
            self._log_contribution_scores(users, scores, raw_values, per_user_outlier_info, None)
        else:
            log("round_scoring", "not using mad")
            scores = calc_contribution_scores_dotproduct(local_updates, global_update)
            self._log_contribution_scores(users, scores, None, None, None)

        return scores


    def _calculate_scores_naive(self, users):
        """
        Equal-share scoring: everyone contributing gets 1 / num_mergers.
        """  # unused; included for signature consistency
        num_mergers = len(users)
        scores = [calc_contribution_score_naive(num_mergers) for _ in users]

        self._log_contribution_scores(users, scores, None, None, None)

        return scores


    def _calculate_scores_accuracy_loss(self, users, mad_threshold = 1.1):
        """
        Accuracy-Loss-based scoring: use accuracy and loss directly as contribution score.
        """
        if self.use_outlier_detection:
            msg = "accuracy_loss strategy does not support MAD outlier detection — outlier_info will not be logged."
            log("round_scoring", yellow(f"WARNING: {msg}"))
            self._log_warning(msg)
        # accuracies: 1d array
        # losses: 1d array
        # prev_acc, prev_loss: int

        # Array of previous accuracies and losses from all users: A tuple of arrays
        prev_accuracies, prev_losses = self.contract.functions.getAllPreviousAccuraciesAndLosses().call()

        # use mad on these and average them

        mad_prev_accuracies = remove_outliers_mad(prev_accuracies, mad_threshold)
        mad_prev_losses = remove_outliers_mad(prev_losses, mad_threshold)

        avg_prev_acc = np.mean(mad_prev_accuracies)
        avg_prev_loss = np.mean(mad_prev_losses)

        avg_accuracies = [] # after loop: [30, 20, 30, 40]
        avg_losses = [] # after loop: [60, 70, 50, 80]

        label_w = max((len(u.display_label()) for u in users), default=12)
        log("round_scoring",f"\naccuracy_loss: loss_baseline={avg_prev_loss:.6f}  acc_baseline={avg_prev_acc:.6f}  (punished at ANY loss worsening)")

        for u in users: # For loop to extract accuracies and loses.

            # All accuracies and loses per user
            _, accuracies, losses = self.contract.functions.getAllAccuraciesLossesAbout(u.address).call()

            try:
                # Multiple accuracies and losses per user
                mad_accuracies = remove_outliers_mad(accuracies, mad_threshold)
                mad_losses = remove_outliers_mad(losses, mad_threshold)

                # One average accuracy and loss per user
                avg_acc = np.mean(mad_accuracies)
                avg_loss = np.mean(mad_losses)

                avg_accuracies.append(avg_acc) # int
                avg_losses.append(avg_loss) # int
                loss_pct_diff = (avg_loss - avg_prev_loss) / avg_prev_loss * 100 if avg_prev_loss != 0 else float('inf')
                loss_status = "PUNISHED" if avg_loss > avg_prev_loss else ("improved" if avg_loss < avg_prev_loss else "neutral")
                log("round_scoring",f"  [{u.display_label():<{label_w}}] loss={avg_loss:.6f}  baseline={avg_prev_loss:.6f}  diff={loss_pct_diff:+.2f}%  [{loss_status}]")
            except ValueError as e:
                log("round_scoring", f"  [{u.display_label():<{label_w}}] SKIPPED ({e}) — using baseline as neutral fallback")
                avg_accuracies.append(avg_prev_acc)
                avg_losses.append(avg_prev_loss)


        scores = []

        norm_accuracies = normalize_contribution_scores_old(avg_accuracies, avg_prev_acc)
        log("round_scoring", f"normalized accuracies: {fmt_floats(norm_accuracies)}")

        norm_losses = normalize_contribution_scores_old(avg_losses, avg_prev_loss)
        log("round_scoring", f"normalized losses:     {fmt_floats(norm_losses)}")

        sum_na = sum(norm_accuracies)
        sum_nl = sum(norm_losses)

        for i in range(len(norm_accuracies)):
            res = (norm_accuracies[i] + norm_losses[i]) / (sum_na + sum_nl)
            score = int(Decimal(res) * Decimal('1e18'))
            scores.append(score)

        log("round_scoring", f"scores (1e18-scaled):  {fmt_scaled_scores(scores)}")
        self._log_contribution_scores(users, scores, raw_values=None, outlier_info=None, previous_avg=None)
        return scores
    # Output: An array of user scores
    # Find out who was merged


    def _calculate_scores_accuracy_only(self, users, mad_threshold = 1.1):
        """
        Accuracy-based scoring: use accuracy directly as contribution score.
        """
        # accuracies: 1d array
        # prev_acc: int


        # Array of previous accuracies from all users: A tuple of arrays
        prev_accuracies, _ = self.contract.functions.getAllPreviousAccuraciesAndLosses().call()

        # use mad on these and average them
        prev_info = {}
        mad_prev_accuracies = remove_outliers_mad(prev_accuracies, mad_threshold, collector=prev_info, label="previous")
        avg_prev_acc = np.mean(mad_prev_accuracies)
        avg_accuracies = [] # after loop: [30, 20, 30, 40]
        per_user_outlier_info = []
        label_w = max((len(u.display_label()) for u in users), default=12)

        for u in users: # For loop to extract accuracies.
            # All accuracies per user
            _, accuracies = self.contract.functions.getAllAccuraciesAbout(u.address).call()

            try:
                # Multiple accuracies per user
                info = {}
                mad_accuracies = remove_outliers_mad(accuracies, mad_threshold, collector=info, label="current")
                # One average accuracy per user
                avg_acc = np.mean(mad_accuracies)
                avg_accuracies.append(avg_acc) # int
                per_user_outlier_info.append({**prev_info, **info}) # Merge prev (global baseline) and current (per-user) MAD info into one dict; keys are prefixed ("previous_*" / "current_*") so they don't collide
            except ValueError as e:
                log("round_scoring", f"  [{u.display_label():<{label_w}}] SKIPPED ({e}) — using baseline as neutral fallback")
                avg_accuracies.append(avg_prev_acc)
                per_user_outlier_info.append({})

        norm_accuracies = normalize_contribution_scores_new(avg_accuracies, avg_prev_acc, 'accuracy')
        log("round_scoring", f"normalized accuracies: {fmt_floats(norm_accuracies)}")

        # Validating Shapley Axioms (Runtime Guard)
        diffs = [v - avg_prev_acc for v in avg_accuracies]
        success, errors = check_shapley_compliance(diffs, norm_accuracies)

        if not success:
            msg = f"[Round {self.pytorch_model.round}] Axiom Violation: {errors}"
            runtime_warnings.append(msg)
            log("round_scoring", colored(f"{msg}", "yellow"))
            self._log_warning(msg)

        scores = [int(Decimal(norm_accuracy_score) * Decimal('1e18')) for norm_accuracy_score in norm_accuracies]
        log("round_scoring", f"scores (1e18-scaled):  {fmt_scaled_scores(scores)}")

        self._log_contribution_scores(users, scores, avg_accuracies, per_user_outlier_info, avg_prev_acc)

        return scores


    def _calculate_scores_loss_only(self, users, mad_threshold = 1.1):
        """
        Loss-based scoring: use loss directly as contribution score.
        """
        # losses: 1d array
        # prev_loss: int

        # Array of previous losses from all users: A tuple of arrays
        _, prev_losses = self.contract.functions.getAllPreviousAccuraciesAndLosses().call()

        # use mad on these and average them
        prev_info = {}
        mad_prev_losses = remove_outliers_mad(prev_losses, mad_threshold, collector=prev_info, label="previous")
        avg_prev_loss = np.mean(mad_prev_losses)
        avg_losses = [] # after loop: [60, 70, 50, 80]
        per_user_outlier_info = []
        label_w = max((len(u.display_label()) for u in users), default=12)

        log("round_scoring",f"\nloss_only: baseline={avg_prev_loss:.6f}  (punished at ANY worsening above baseline)")

        for u in users: # For loop to extract losses.
            # All loses per user
            _, losses = self.contract.functions.getAllLossesAbout(u.address).call()

            try:
                # Multiple accuracies and losses per user
                info = {}
                mad_losses = remove_outliers_mad(losses, mad_threshold, collector=info, label="current")
                # One average accuracy and loss per user
                avg_loss = np.mean(mad_losses)
                avg_losses.append(avg_loss) # int
                per_user_outlier_info.append({**prev_info, **info}) # Merge prev (global baseline) and current (per-user) MAD info into one dict; keys are prefixed ("previous_*" / "current_*") so they don't collide
                pct_diff = (avg_loss - avg_prev_loss) / avg_prev_loss * 100 if avg_prev_loss != 0 else float('inf')
                ustatus = "PUNISHED" if avg_loss > avg_prev_loss else ("improved" if avg_loss < avg_prev_loss else "neutral")
                log("round_scoring",f"  [{u.display_label():<{label_w}}] loss={avg_loss:.6f}  baseline={avg_prev_loss:.6f}  diff={pct_diff:+.2f}%  [{ustatus}]")
            except ValueError as e:
                log("round_scoring", f"  [{u.display_label():<{label_w}}] SKIPPED ({e}) raw_losses={losses} — using baseline as neutral fallback")
                avg_losses.append(avg_prev_loss)
                per_user_outlier_info.append({})

        norm_losses = normalize_contribution_scores_new(avg_losses, avg_prev_loss, 'loss')
        log("round_scoring", f"normalized losses: {fmt_floats(norm_losses)}")

        # Validating Shapley Axioms (Runtime Guard)
        diffs = [v - avg_prev_loss for v in avg_losses]
        diffs = [-1 * d for d in diffs]
        success, errors = check_shapley_compliance(diffs, norm_losses)

        if not success:
            msg = f"[Round {self.pytorch_model.round}] Axiom Violation: {errors}"
            runtime_warnings.append(msg)
            log("round_scoring", colored(f"{msg}", "yellow"))
            self._log_warning(msg)

        scores = [int(Decimal(norm_accuracy_score) * Decimal('1e18')) for norm_accuracy_score in norm_losses]

        log("round_scoring", f"scores (1e18-scaled):  {fmt_scaled_scores(scores)}")

        self._log_contribution_scores(users, scores, avg_losses, per_user_outlier_info, avg_prev_loss)

        return scores


    def _calculate_scores_loss_tolerance_aware(self, users, mad_threshold=1.1):
        # Loss-based scoring with shifted reward threshold.
        # Reward zone moves from `loss < avg_prev_loss` to `loss < avg_prev_loss + ε`,
        # where ε = loss_tolerance_pct * avg_prev_loss. Small worsenings still receive
        # a small positive contribution; improvements remain ranked above them.
        _, prev_losses = self.contract.functions.getAllPreviousAccuraciesAndLosses().call()

        prev_info = {}
        mad_prev_losses = remove_outliers_mad(prev_losses, mad_threshold, collector=prev_info, label="previous")
        avg_prev_loss = float(np.mean(mad_prev_losses))

        epsilon = self.loss_tolerance_pct * avg_prev_loss
        shifted_baseline = avg_prev_loss + epsilon

        avg_losses = []
        per_user_outlier_info = []
        label_w = max((len(u.display_label()) for u in users), default=12)

        log("round_scoring", f"\nloss_tolerance_aware: baseline={avg_prev_loss:.6f}  ε={epsilon:.6f} ({self.loss_tolerance_pct*100:.1f}% of baseline)  punish_threshold={shifted_baseline:.6f}")

        for u in users:
            _, losses = self.contract.functions.getAllLossesAbout(u.address).call()
            try:
                info = {}
                mad_losses = remove_outliers_mad(losses, mad_threshold, collector=info, label="current")
                avg_loss = float(np.mean(mad_losses))
                avg_losses.append(avg_loss)
                raw_diff = avg_loss - avg_prev_loss
                in_band = 0.0 < raw_diff <= epsilon
                per_user_outlier_info.append({
                    **prev_info,
                    **info,
                    "tolerance_strategy": "loss_tolerance_aware",
                    "tolerance_epsilon": epsilon,
                    "tolerance_baseline_shifted": shifted_baseline,
                    "tolerance_raw_diff": raw_diff,
                    "tolerance_in_band": in_band,
                })
                pct_diff = raw_diff / avg_prev_loss * 100 if avg_prev_loss != 0 else float('inf')
                margin = shifted_baseline - avg_loss
                pct_margin = margin / avg_prev_loss * 100 if avg_prev_loss != 0 else float('inf')
                if raw_diff <= 0:
                    ustatus = "improved"
                elif in_band:
                    ustatus = "in-band (safe)"
                else:
                    ustatus = "PUNISHED (beyond ε)"
                log("round_scoring",f"  [{u.display_label():<{label_w}}] loss={avg_loss:.6f}  baseline={avg_prev_loss:.6f}  diff={pct_diff:+.2f}%  margin_to_punish={pct_margin:+.2f}% ({margin:+.6f})  [{ustatus}]")
            except ValueError as e:
                log("round_scoring", f"  [{u.display_label():<{label_w}}] SKIPPED ({e}) raw_losses={losses} — using baseline as neutral fallback")
                avg_losses.append(avg_prev_loss)
                per_user_outlier_info.append({})

        norm_losses = normalize_contribution_scores_new(avg_losses, shifted_baseline, 'loss')
        log("round_scoring", f"normalized losses: {fmt_floats(norm_losses)}")

        diffs = [-(v - shifted_baseline) for v in avg_losses]
        success, errors = check_shapley_compliance(diffs, norm_losses)

        if not success:
            msg = f"[Round {self.pytorch_model.round}] Axiom Violation: {errors}"
            runtime_warnings.append(msg)
            log("round_scoring", colored(f"{msg}", "yellow"))
            self._log_warning(msg)

        scores = [int(Decimal(v) * Decimal('1e18')) for v in norm_losses]
        log("round_scoring", f"scores (1e18-scaled):  {fmt_scaled_scores(scores)}")

        self._log_contribution_scores(users, scores, avg_losses, per_user_outlier_info, shifted_baseline)
        return scores


    def _calculate_scores_loss_tolerance_snap(self, users, mad_threshold=1.1):
        # Loss-based scoring with snap-to-baseline for small worsenings.
        # If a user's avg_loss is within ε above avg_prev_loss, snap it to avg_prev_loss
        # so the diff is zero (treated as neutral). Improvements untouched.
        # Big worsenings beyond ε still penalized.
        _, prev_losses = self.contract.functions.getAllPreviousAccuraciesAndLosses().call()

        prev_info = {}
        mad_prev_losses = remove_outliers_mad(prev_losses, mad_threshold, collector=prev_info, label="previous")
        avg_prev_loss = float(np.mean(mad_prev_losses))

        epsilon = self.loss_tolerance_pct * avg_prev_loss
        snap_threshold = avg_prev_loss + epsilon

        avg_losses = []
        per_user_outlier_info = []
        label_w = max((len(u.display_label()) for u in users), default=12)

        log("round_scoring", f"\nloss_tolerance_snap: baseline={avg_prev_loss:.6f}  ε={epsilon:.6f} ({self.loss_tolerance_pct*100:.1f}% of baseline)  snap_threshold={snap_threshold:.6f}")

        for u in users:
            _, losses = self.contract.functions.getAllLossesAbout(u.address).call()
            try:
                info = {}
                mad_losses = remove_outliers_mad(losses, mad_threshold, collector=info, label="current")
                raw_avg_loss = float(np.mean(mad_losses))
                raw_diff = raw_avg_loss - avg_prev_loss
                in_band = 0.0 < raw_diff <= epsilon
                snapped_avg_loss = avg_prev_loss if in_band else raw_avg_loss
                avg_losses.append(snapped_avg_loss)
                per_user_outlier_info.append({
                    **prev_info,
                    **info,
                    "tolerance_strategy": "loss_tolerance_snap",
                    "tolerance_epsilon": epsilon,
                    "tolerance_raw_avg_loss": raw_avg_loss,
                    "tolerance_raw_diff": raw_diff,
                    "tolerance_in_band": in_band,
                    "tolerance_snapped": in_band,
                })
                pct_diff = raw_diff / avg_prev_loss * 100 if avg_prev_loss != 0 else float('inf')
                margin = epsilon - raw_diff
                pct_margin = margin / avg_prev_loss * 100 if avg_prev_loss != 0 else float('inf')
                if raw_diff <= 0:
                    ustatus = "improved"
                elif in_band:
                    ustatus = "SNAPPED to neutral (within ε)"
                else:
                    ustatus = "PUNISHED (beyond ε)"
                log("round_scoring", f"  [{u.display_label():<{label_w}}] raw_loss={raw_avg_loss:.6f}  eff_loss={snapped_avg_loss:.6f}  baseline={avg_prev_loss:.6f}  diff={pct_diff:+.2f}%  margin_to_threshold={pct_margin:+.2f}% ({margin:+.6f})  [{ustatus}]")
            except ValueError as e:
                log("round_scoring", f"  [{u.display_label():<{label_w}}] SKIPPED ({e}) raw_losses={losses} — using baseline as neutral fallback")
                avg_losses.append(avg_prev_loss)
                per_user_outlier_info.append({})

        norm_losses = normalize_contribution_scores_new(avg_losses, avg_prev_loss, 'loss')
        log("round_scoring", f"normalized losses: {fmt_floats(norm_losses)}")

        diffs = [-(v - avg_prev_loss) for v in avg_losses]
        success, errors = check_shapley_compliance(diffs, norm_losses)

        if not success:
            msg = f"[Round {self.pytorch_model.round}] Axiom Violation: {errors}"
            runtime_warnings.append(msg)
            log("round_scoring", colored(f"{msg}", "yellow"))
            self._log_warning(msg)

        scores = [int(Decimal(v) * Decimal('1e18')) for v in norm_losses]
        log("round_scoring", f"scores (1e18-scaled):  {fmt_scaled_scores(scores)}")

        self._log_contribution_scores(users, scores, avg_losses, per_user_outlier_info, avg_prev_loss)
        return scores


    def trim_global_update_using_mad(self,
                                     local_updates: torch.Tensor,
                                     global_update: torch.Tensor,
                                     mad_thresh: float = 3.5,
                                     eps: float = 1e-12):
        """
        Trim the global update by removing (zeroing) weights where
        all clients are outliers according to MAD filtering.

        Args:
            local_updates: Tensor (num_mergers, D)
            global_update: Tensor (D,)
            mad_thresh: MAD robust z-score threshold
            eps: avoid divide-by-zero

        Returns:
            filtered_global_update: Tensor (D,)
            per_user_outlier_info: list of dicts (one per user) with MAD stats
        """

        num_mergers, D = local_updates.shape

        # Per-weight median
        median = local_updates.median(dim=0).values  # (D,)

        # Per-weight absolute deviation
        abs_dev = (local_updates - median).abs()  # (num_mergers, D)

        # MAD per weight
        mad = abs_dev.median(dim=0).values  # (D,)
        safe_mad = mad.clone()
        safe_mad[safe_mad < eps] = eps

        # Per weight/user robust z-score
        robust_z = 0.6745 * abs_dev / safe_mad

        # Non-outlier mask (True = keep)
        mask = robust_z <= mad_thresh  # (num_mergers, D)

        # Collapse user dimension: keep weight if ANY user is non-outlier
        global_mask = mask.any(dim=0)  # (D,)

        # Zero out outlier-only weights in global update
        filtered_global_update = global_update * global_mask

        # Per-user summary stats for logging
        mad_mean = float(mad.mean().item())
        median_mean = float(median.mean().item())
        per_user_outlier_info = [
            {
                "current_median": median_mean,
                "current_mad": mad_mean,
                "current_boundary": mad_thresh,
                # Weight-space outlier counts (not scalar value lists — stored under distinct keys)
                "dotproduct_outlier_weight_count": int((~mask[i]).sum().item()),
                "dotproduct_outlier_weight_fraction": float((~mask[i]).float().mean().item()),
            }
            for i in range(num_mergers)
        ]

        return filtered_global_update, per_user_outlier_info


    def get_round_rewards(self, receipt):
        events = self.get_events(
            receipt=receipt,
            event_names=["Reward"]
        )
        reward_events = events["Reward"]
        
        result = []
        for ev in reward_events:
            if ev["roundScore"] > 0:
                result.append(
                    (
                        ev["user"],
                        ev["roundScore"],
                        ev["win"], # Reward/Punishment
                        ev["newReputation"], # New global reputation after reward/punishment
                        ev["is_reward"] # Boolean
                    )
                )
        return result



    # ---- logging helpers ----

    def _log_receipt(self, receipt, receipt_type, round=None):  # delegates to ExperimentLogger
        if self._logger is None:
            return
        self._logger.log_receipt(
            round=self.pytorch_model.round if round is None else round,
            tx_type=receipt_type,
            tx_hash=receipt["transactionHash"].hex(),
            gas_used=receipt["gasUsed"],
        )

    def _log_warning(self, msg):
        if self._logger is None:
            return
        self._logger.log_warning(self.pytorch_model.round, msg)

    def _log_contribution_scores(self, users, scores, raw_values, outlier_info, previous_avg):
        if self._logger is None:
            return
        self._logger.log_contribution_scores(
            round=self.pytorch_model.round,
            user_numbers=[u.number for u in users],
            user_addresses=[u.address for u in users],
            scores=scores,
            raw_values=raw_values,
            outlier_info=outlier_info,
            previous_avg=previous_avg,
        )

    def _log_round_zero(self):
        if self._logger is None:
            return
        self._logger.log_global_round(
            round=0,
            round_time=0.0,
            obj_global_acc=self.pytorch_model.accuracy[-1] if self.pytorch_model.accuracy else None,
            obj_global_loss=self.pytorch_model.loss[-1]    if self.pytorch_model.loss     else None,
            reward_pool=self._reward_balance[-1],
            punishment_pool=0,
        )
        all_participants = self.pytorch_model.participants + self.pytorch_model.disqualified
        for _participant in all_participants:
            self._logger.log_user_round(
                round=0,
                user_number=_participant.number,
                state="active",
                behavior=_participant.attitude,
                role=_participant.futureAttitude,
                grs=_participant._globalrep[-1],
                address=_participant.address,
                id=_participant.id,
                sub_personal_acc=None,
                sub_personal_loss=None,
                sub_global_acc=None,
                sub_global_loss=None,
                round_reputation_assigned=None,
                reward_delta=None,
                is_reward=None,
                merged=None,
            )

    def _log_global_round(self, round, round_time, punishment_pool):
        if self._logger is None:
            return
        self._logger.log_global_round(
            round=round,
            round_time=round_time,
            obj_global_acc=self.pytorch_model.accuracy[-1] if self.pytorch_model.accuracy else 0,
            obj_global_loss=self.pytorch_model.loss[-1] if self.pytorch_model.loss else 0,
            reward_pool=self._reward_balance[-1],
            punishment_pool=punishment_pool,
        )

    def _log_round(self, current_round, round_time,
                   evaluationData: EvaluationData,
                   contributors, receipt):
        if self._logger is None:
            return


        # ---- votes ----
        for _giver in self.pytorch_model.participants:
            userData = evaluationData.get(_giver.id)
            _user_acc  = userData.prev_accuracy if userData.prev_accuracy else None
            _user_loss = userData.prev_loss  if userData.prev_loss  else None
            for _receiver in self.pytorch_model.participants:
                if _giver.address == _receiver.address:
                    continue
                try:
                    _feedback_vote = userData.feedback[_receiver.id]
                except (IndexError, TypeError):
                    continue
                self._logger.log_vote(
                    round=current_round,
                    giver_address=_giver.address,
                    giver_id=_giver.id,
                    receiver_address=_receiver.address,
                    receiver_id=_receiver.id,
                    vote_feedback_score=_feedback_vote,
                    vote_prev_accuracy=_user_acc,
                    vote_prev_loss=_user_loss,
                    vote_accuracy=userData.accuracy[_receiver.id] if userData.accuracy is not None else None,
                    vote_loss=userData.loss[_receiver.id]         if userData.loss     is not None else None,
                )

        # ---- per-user round ----
        _round_rewards  = self.get_round_rewards(receipt) if receipt is not None else []
        _addr_to_reward = {addr: win for addr, _rs, win, _nr, _ir in _round_rewards}
        _addr_to_ir     = {addr: _ir  for addr, _rs, win, _nr, _ir in _round_rewards}

        for _user in self.pytorch_model.participants:
            self._logger.log_user_round(
                round=current_round, user_number=_user.number, state="active",
                behavior=_user.attitude, role=_user.futureAttitude,
                grs=_user._globalrep[-1],
                sub_personal_acc=_user.currentAcc,
                sub_personal_loss=_user.currentLoss,
                sub_global_acc=_user._accuracy[-1],
                sub_global_loss=_user._loss[-1],
                id = _user.id,
                round_reputation_assigned=_user._roundrep[-1] if _user._roundrep else None,
                reward_delta=_addr_to_reward.get(_user.address, None),
                is_reward=_addr_to_ir.get(_user.address, None),
                merged=any(u.address == _user.address for u in contributors),
            )
        for _user in self.pytorch_model.disqualified:
            self._logger.log_user_round(
                round=current_round, user_number=_user.number, state="disqualified",
                behavior=_user.attitude, role=_user.futureAttitude,
                grs=_user._globalrep[-1],
                sub_personal_acc=_user.currentAcc,
                sub_personal_loss=_user.currentLoss,
                sub_global_acc=_user._accuracy[-1],
                sub_global_loss=_user._loss[-1],
                id=_user.id,
                round_reputation_assigned=_user._roundrep[-1] if _user._roundrep else None,
                reward_delta=_addr_to_reward.get(_user.address, None),
                is_reward=_addr_to_ir.get(_user.address, None),
                merged=False,
            )

        # ---- global round ----
        _punishment_total = sum(p[1] for p in self._punishments if p[0] == current_round)
        self._log_global_round(current_round, round_time, _punishment_total)



    def simulate(self, rounds):
        """
        Run a full FL simulation for a given number of rounds.
        High-level flow per round:
          1) Update user attitudes
          2) Local training
          3) Let malicious/freerider users modify/copy models
          4) Register slots & provide hashed weights
          5) Exchange and verify models
          6) Evaluation & feedback
          7) Merge models
          8) Compute contribution scores
          9) Close round, print summary
        At the end, all users exit the system.
        """
        log("setup_contracts", self.contractAddress)
        #self.let_users_join()
        flags = globals.ReplayMode._actively_replaying | globals.ReplayMode.HardPlayBack
        if (globals.reuse_runs & flags) == flags:
            return self.pytorch_model.runRepo.get_task_rep_delta_and_GRS(-1, "get_task_rep_delta_and_GRS-simulate", self.contract, self.pytorch_model.get_participant)
        
        grs = [(participant.address, participant._globalrep[-1]) for participant in self.pytorch_model.participants + self.pytorch_model.disqualified]
        
        roundTx = self.txHashes[self.writeTxProgress:]
        self.writeTxProgress = len(self.txHashes)

        self.writer.writeResult({
                "round": 0,
                "GRS": grs,
                "globalAcc": self.pytorch_model.accuracy[-1] or 0,
                "globalLoss": self.pytorch_model.loss[-1] or 0,
                "conctractBalanceRewards": self._reward_balance[-1],
                "round_rewards": [],
                "accAvgPerUser": [],
                "lossAvgPerUser": [],
                "feedbackMatrix": None,
                "disqualifiedUsers": [],
                "contributionScores": [],
                "userStatuses": [user.get_status() for user in self.pytorch_model.participants],
                "GasTransactions": roundTx
            })

        self._log_round_zero()

        def _sample_free_ram_gb():
            try:
                import psutil
                return psutil.virtual_memory().available / (1024 ** 3)
            except Exception:
                try:
                    with open("/proc/meminfo") as _f:
                        for _line in _f:
                            if _line.startswith("MemAvailable:"):
                                return int(_line.split()[1]) / (1024 * 1024)
                except Exception:
                    pass
            return None

        globals.min_free_ram_gb = None
        _min_free_gb = float('inf')

        for roundnr in range(rounds):
            log("round_boundary", b(f"Round {self.pytorch_model.round} starts..."))
            _round_start = time.perf_counter()
            self.pytorch_model.update_users_attitude()

            self.pytorch_model.federated_training()

            self.pytorch_model.let_malicious_users_do_their_work()
            self.pytorch_model.let_freerider_users_do_their_work()
            
            self.users_register_slot()

            self.users_provide_hashed_weights()

            self.pytorch_model.exchange_models()

            on_chain_hashed_weights = self.pytorch_model.runRepo.on_chain_hashed_weights(self.pytorch_model.round, f"on_chain_hashed_weights-simulate", self)

            self.pytorch_model.verify_models(on_chain_hashed_weights)

            self.evaluation: EvaluationData = self.pytorch_model.evaluation()

            self.quick_feedback_round(self.evaluation)

            # A roundRep of 0, does not nec. mean mal.
            contributors = [user for user in self.pytorch_model.participants if user._roundrep[-1] >= 0] # Keeps track of who will be merged in the_merge()
            self.pytorch_model.the_merge(contributors)

            log("round_boundary", b("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n"))
            self.contribution_score(contributors)
            receipt = self.close_round()

            log("round_boundary", b(f"Round {self.pytorch_model.round - 1} actually completed:"))
            _all_users = self.pytorch_model.participants + self.pytorch_model.disqualified
            _lw = max((len(u.display_label()) for u in _all_users), default=12)
            for user in _all_users:
                user._globalrep.append(self.get_global_reputation_of_user(user.address))
                i, j = user._globalrep[-2:]
                log("round_boundary", b("{:<{lw}} ({})  {:>25,.0f} -> {:>25,.0f}".format(user.display_label(), user.address[0:16] + "...", i, j, lw=_lw)))

            # self.print_round_summary(receipt)
            if receipt is not None:
                self.print_round_summary(receipt)

            _round_time = time.perf_counter() - _round_start
            _current_round = self.pytorch_model.round - 1

            self._log_round(
                _current_round, _round_time,
                self.evaluation,
                contributors, receipt,
            )

            grs = [(user.address, user._globalrep[-1]) for user in self.pytorch_model.participants + self.pytorch_model.disqualified]
            round_punishment = [(punishment[0], punishment[1]) for punishment in self._punishments if punishment[0] == self.pytorch_model.round - 1]
            round_kicked = [punishment[2] for punishment in self._punishments if punishment[0] == self.pytorch_model.round - 1]
            roundTx = self.txHashes[self.writeTxProgress:]
            self.writeTxProgress = len(self.txHashes) - 1
            self.writer.writeResult({
                "round": self.pytorch_model.round - 1,
                "GRS": grs,
                "globalAcc": self.pytorch_model.accuracy[-1] or 0, # Checks out
                "globalLoss": self.pytorch_model.loss[-1] or 0, # Checks out
                "conctractBalanceRewards": self._reward_balance[-1],
                "round_punishments": round_punishment,
                "round_rewards": self.get_round_rewards(receipt),
                "accAvgPerUser": self.evaluation.prev_accuracies, # Check - Should come from am
                "lossAvgPerUser": self.evaluation.prev_losses, # Check - Should come from lm
                "feedbackMatrix": self.evaluation.feedback_matrix,
                "disqualifiedUsers": round_kicked,
                "contributionScores": self.scores,
                "userStatuses": [user.get_status() for user in self.pytorch_model.participants],
                "GasTransactions": roundTx,
                })
            globals.progress = int(
                ((roundnr + 1) / rounds) * 100
            )
            log("round_scoring",f"round progress: {globals.progress}")

            _free = _sample_free_ram_gb()
            if _free is not None:
                _min_free_gb = min(_min_free_gb, _free)
                globals.min_free_ram_gb = _min_free_gb
                log("round_boundary", f"[mem] round {self.pytorch_model.round - 1}: free={_free:.2f} GB  peak_min_free={_min_free_gb:.2f} GB")

        log("round_scoring", f"Number of Shapley Axioms violated: {len(runtime_warnings)}\n")
        if runtime_warnings:
            log("round_scoring", "\n" + red("!" * 30 + " SHAPLEY WARNINGS " + "!" * 30))
            for warn in runtime_warnings:
                log("round_scoring", colored(warn, 'yellow'))
            log("round_scoring", red("!" * 78))
        if self.writer is not None:
            self.writer.write_comment(f"$gasCosts${self.gas_feedback},{self.gas_register},{self.gas_slot},{self.gas_weights},{self.gas_close},{self.gas_deploy},{self.gas_exit}")
        trs = self.pytorch_model.runRepo.get_task_rep_delta_and_GRS(-1, "get_task_rep_delta_and_GRS-simulate", self.contract, self.pytorch_model.get_participant)
        self.writer.write_comment(f"$trs${trs}")
        self._logger.log_trs(trs)
        self.pytorch_model.runRepo.flush()

        self._finalize_reputations()

        return trs
            
            
    
    def visualize_simulation(self, output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)  # ensure folder exists
        accuracy = [0] + self.pytorch_model.accuracy
        loss = [self.pytorch_model.loss[0]] + self.pytorch_model.loss

        f, axs = plt.subplots(1, 4,figsize=(16, 3),  gridspec_kw={'width_ratios': [0.8,2,2,2],
                                                                      'height_ratios': [1]})
        colors = ["#00629b", "#629b00", "#000000", "#d93e6a"]

        participants =self.pytorch_model.participants + self.pytorch_model.disqualified

        #  True to get old setep graph, False to get point graph
        use_step_grs = False

        rounds = list(range(len(accuracy)))
        #x = [item for sublist in zip(x,(np.array(x)+1).tolist()) for item in sublist]

        y = accuracy
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        acc_line = axs[1].plot(rounds, y, color=colors[0], linewidth=2.5, label="Avg. Accuracy")[0]
        twin = axs[1].twinx()
        y = loss
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        loss_line = twin.plot(rounds, y, color=colors[1], linewidth=2.5, linestyle="--", label="Avg. Loss")[0]

        grs_rounds = list(range(len(participants[0]._globalrep)))
        if use_step_grs:
            grs_x = [item for sublist in zip(grs_rounds, (np.array(grs_rounds) + 1).tolist()) for item in sublist]
            grs_ticks = grs_rounds
            for i, user in enumerate(participants):
                grs_y = [item for sublist in zip(user._globalrep, user._globalrep) for item in sublist]
                axs[2].plot(grs_x, grs_y, linewidth=2.5, color=user.color)
        else:
            grs_x = grs_rounds
            grs_ticks = grs_rounds
            # plotting the points
            for i, user in enumerate(participants):
                axs[2].plot(
                    grs_x,
                    user._globalrep,
                    linewidth=2.5,
                    color=user.color,
                    alpha=0.9,
                    marker="o",
                    markersize=4,
                    markevery=1,
                )

        pun = {}
        for i, j, y in self._punishments:
            if i in pun.keys():
                pun[i] += j
            else:
                pun[i] = j

        rew = list()
        for i, j in enumerate(self._reward_balance):
            if i in pun.keys():
                rew.append(j+pun[i])
            else:
                rew.append(j)    

        y_reward = [item for sublist in zip(self._reward_balance,self._reward_balance) for item in sublist]
        y2_reward = [item for sublist in zip(rew,rew) for item in sublist]
        x_reward = list(range(len(self._reward_balance)))
        x_reward = [item for sublist in zip(x_reward,(np.array(x_reward)+1).tolist()) for item in sublist]

        axs[3].plot(x_reward,y_reward, label="reward", color=colors[0], linewidth=2.5)
        axs[3].plot(x_reward,y2_reward, label="reward +\npunishments", color=colors[1], linewidth=2.5)
        axs[3].fill_between(x_reward,y_reward, y2_reward, alpha=0.2, hatch=r"//", color = colors[1])


        axs[0].text(0, 0.1, f'dataset={self.pytorch_model.config.dataset}\n'\
                                 + f'epochs={self.pytorch_model.config.epochs}\n' \
                                 + f'rounds={self.pytorch_model.round-1}\n' \
                                 + f'users={self.pytorch_model.NUMBER_OF_CONTRIBUTERS}\n' \
                                 + f'malicious={self.pytorch_model.NUMBER_OF_BAD_CONTRIBUTORS}\n'\
                                 + f'copycat={self.pytorch_model.NUMBER_OF_FREERIDER_CONTRIBUTORS}', fontsize=15)
        axs[0].set_axis_off()
        
        axs[1].set_xlabel('rounds\n(a)', fontsize=14)
        axs[2].set_xlabel('rounds\n(b)', fontsize=14)
        axs[3].set_xlabel('rounds\n(c)', fontsize=14)
        #axs[0].set_ylabel(f'users={participants};\n malicious={malicious_users};\n copycat={sneaky_freerider}', fontsize=14)
        axs[1].set_ylabel('Avg. Accuracy', fontsize=14)
        twin.set_ylabel('Avg. Loss', fontsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

        axs[2].set_ylabel('GRS', fontsize=14)
        axs[3].set_ylabel('Contract Balance', fontsize=14)

        axs[2].tick_params(axis='both', which='major', labelsize=14)
        axs[3].tick_params(axis='both', which='major', labelsize=14)
        
        if len(rounds) > 20:
            axs[1].set_xticks([i for i in rounds if i%5==0 or i == 0])
        else:
            axs[1].set_xticks([i for i in rounds])

        if len(grs_ticks) > 20:
            axs[2].set_xticks([i for i in grs_ticks if i%5==0 or i == 0])
        else:
            axs[2].set_xticks([i for i in grs_ticks])

        if len(x_reward) > 20:
            axs[3].set_xticks([i for i in x_reward if i%5==0 or i == 0])
        else:
            axs[3].set_xticks([i for i in x_reward])
    
        axs[1].set_xlim(0, max(rounds) if rounds else 0)
        
        axs[2].yaxis.get_offset_text().set_fontsize(14)
        axs[3].yaxis.get_offset_text().set_fontsize(14)
        
        axs[1].grid()
        axs[2].grid()
        axs[3].grid()

        # Legend for the dual-axis accuracy/loss plot
        twin_lines = [acc_line, loss_line]
        axs[1].legend(twin_lines, [l.get_label() for l in twin_lines], loc="lower right", fontsize=10)

        lgnd = axs[3].legend(fontsize=10)

        # giving a title to my graph 
        #axs[1].set_title(f'users={participants}; malicious={malicious_users}; copycat={sneaky_freerider}', fontsize=10) 

        # function to show the plot
        log("experiment_end", output_folder_path)
        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(output_folder_path, f"{self.pytorch_model.config.dataset}_simulation.pdf"), bbox_inches='tight')
        #plt.show()
        return plt
    
    def let_users_join(self, JobListing: JobListing):
        txs = JobListing.let_all_participants_register(self.pytorch_model.participants)
        
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar("setup_contracts", i, l)
            receipt = globals.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_register.append(receipt["gasUsed"])
            self.txHashes.append(("register",receipt["transactionHash"].hex(), receipt["gasUsed"]))
        log("setup_contracts", "-----------------------------------------------------------------------------------")

    def make_participants_from_users(self, users: List[User]):
        users_by_address = {u.address: u for u in users}

        if self.pytorch_model.replaying:
            users = [users_by_address.get(par_addr) for par_addr in users_by_address]
            combinedUsers = self.pytorch_model.runRepo.get_participants(users)
            for user in combinedUsers:
                self.pytorch_model.add_participant(user)
        else:
            for par_addr in self.participant_addresses:
                user = users_by_address.get(par_addr)
                if user is not None:
                    self.pytorch_model.add_participant(user)

        self.pytorch_model.runRepo.set_participants(self.pytorch_model.participants)


def calc_contribution_score_naive(num_mergers) -> int:
    score = Decimal(1) / Decimal(num_mergers)
    return int(score * Decimal('1e18'))

# New function
def calc_contribution_scores_dotproduct(local_updates: torch.Tensor,
                                        global_update: torch.Tensor,
                                        eps: float = 1e-12):
    """
    Compute contribution scores solely using dot-product similarity
    between local updates and the global update.

    Args:
        local_updates: Tensor of shape (num_mergers, D)
                       flattened parameters for each user's local model.
        global_update: Tensor of shape (D,)
                       flattened parameters for the global model.
        eps:           Small tolerance to avoid division by zero.

    Returns:
        List[int]: contribution scores scaled by 1e18.
    """

    num_mergers, D = local_updates.shape

    # ||U||^2
    norm_U_sq = torch.dot(global_update, global_update)

    if norm_U_sq.abs() < eps:
        # If the global update has no magnitude → equal contribution
        score = Decimal(1) / Decimal(num_mergers)
        equal_int_score = int(score * Decimal('1e18'))
        return [equal_int_score for _ in range(num_mergers)]

    # Dot product for each user vs global update
    dots = torch.mv(local_updates, global_update)  # (num_mergers,)
    scores = dots / (num_mergers * norm_U_sq)

    # Convert to integer fixed-point (×1e18)
    return [
        int(Decimal(score.item()) * Decimal('1e18'))
        for score in scores
    ]


def normalize_contribution_scores_old(arr, prev_val):
    # This method takes a 1d array of an array (accuracy or loss), a scalar of previous accuracy or loss
    # Output is an array of normalized (according to sum) input array values
    # Takes a list of values
    # Subtracts a baseline (prev_val)
    # Normalizes them so they sum to 1
    # Example:
    # -- arr - prev_val => norm_arr = [2, 1, 0]
    # -- sum = 3
    # -- [2/3, 1/3, 0/3]

    norm_arr = []
    sum_val = 0.0

    for i in range(len(arr)):
        norm_arr.append(arr[i] - prev_val)
        sum_val += norm_arr[i]

    if len(norm_arr) == 0:
        raise Exception("No values to normalize")
    for i in range(len(norm_arr)):
        if sum_val == 0.0:
            return [1.0 / len(norm_arr)] * len(norm_arr)
        norm_arr[i] /= sum_val
    return norm_arr


def normalize_contribution_scores_new(vals: list, prev_val: float, evaluation_metric: str) -> list:
    """
    4-step normalization for contribution scores.

    1. Subtract baseline, then negate if metric is 'loss' (lower=better → flip sign).
    2. Edge cases: if max==0 replace zeros with 1; if all negative compute sum/val ratios.
    3. Clamp negatives so the minimum is exactly -1.
    4. Final normalization to sum=1: divide by sum if all-positive, otherwise
       redistribute the excess proportionally across positive values.
    """

    def _fmt(xs):
        return "[" + ", ".join(f"{x:.6f}" for x in xs) + "]"

    def _log_step(label, vals, sum_):
        log("round_scoring", f"  {label:<24} sum={sum_:+.6f}  vals={_fmt(vals)}")

    # Step 1: subtract baseline, flip sign for loss
    vals = [v - prev_val for v in vals]
    if evaluation_metric == "loss":
        vals = [-1 * val for val in vals]
    sum_ = sum(vals)
    _log_step("1) shift+sign", vals, sum_)

    # Step 2: edge cases  TODO: 0 og -0.5
    max_val = max(vals)
    if max_val == 0:
        vals = [1 if val == 0 else val for val in vals]
    elif max_val < 0:
        vals = [sum_ / val for val in vals]
    sum_ = sum(vals)
    _log_step("2) edge cases", vals, sum_)

    # Step 3: clamp negatives to minimum -1
    if min(vals) < -1:
        divisor = -min(vals)
        vals = [val / divisor if val < 0 else val for val in vals]
    sum_ = sum(vals)
    _log_step("3) clamp neg >= -1", vals, sum_)

    # Step 4: normalize to sum = 1
    if not sum_ == 1:
        if min(vals) >= 0:
            vals = [val / sum_ for val in vals]
        else:
            sum_of_positives = sum(val for val in vals if val > 0)
            excess_sum = sum_ - 1
            vals = [val + (val / sum_of_positives) * -excess_sum if val > 0 else val for val in vals]
        sum_ = sum(vals)
    _log_step("4) normalize sum=1", vals, sum_)

    return vals


def remove_outliers_mad(arr, threshold=0.70, return_mask=False, collector=None, label=None):
    # Keep original dtype (int from contract uint256). np.median returns float64
    # automatically, so all intermediate MAD arithmetic stays in float without
    # needing to cast the input array.
    arr = np.asarray(arr)

    # always flatten
    flat = arr.ravel()

    if len(flat) == 0:
        raise ValueError("remove_outliers_mad: empty input array")

    median = np.median(flat)
    abs_dev = np.abs(flat - median)
    mad = np.median(abs_dev)

    prefix = f"{label}_" if label else "" # Set label if provided else empty string

    # SPECIAL CASE: MAD == 0
    if mad == 0:
        mask = abs_dev <= threshold
        if collector is not None:
            collector[f"{prefix}median"]   = float(median)
            collector[f"{prefix}mad"]      = 0.0
            collector[f"{prefix}removed"]  = flat[~mask].tolist()
            collector[f"{prefix}accepted"] = flat[mask].tolist()
            collector[f"{prefix}boundary"] = None
        if return_mask:
            return arr, mask
        filtered = flat[mask]
        return filtered if len(filtered) > 0 else flat

    # proper modified z-score
    z_val = 0.6745
    modified_z = z_val * (flat - median) / mad

    mask = np.abs(modified_z) <= threshold

    if collector is not None:
        collector[f"{prefix}median"]   = float(median)
        collector[f"{prefix}mad"]      = float(mad)
        collector[f"{prefix}removed"]  = flat[~mask].tolist()
        collector[f"{prefix}accepted"] = flat[mask].tolist()
        collector[f"{prefix}boundary"] = float(threshold * mad / z_val)

    if return_mask:
        return arr, mask
    filtered = flat[mask]
    # If all values were removed as outliers, fall back to the unfiltered array
    # to prevent np.mean([]) returning NaN downstream.
    return filtered if len(filtered) > 0 else flat

runtime_warnings = []