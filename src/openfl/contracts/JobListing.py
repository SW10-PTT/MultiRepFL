from typing import List
from openfl.api import ConnectionHelper, globals
from openfl.ml.pytorch_model import PytorchModel


class JobListing(ConnectionHelper):
  def __init__(self, address):
    self.contract = self.initialize_job(address)
    pass

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
      