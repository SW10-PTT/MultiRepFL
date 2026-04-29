import os
import platform
import psutil
import time
from pathlib import Path

from web3.exceptions import ContractLogicError
from typing import List
from experiment_configuration import ExperimentConfiguration
from openfl.contracts.JobListing import JobListing
from openfl.ml import pytorch_model as PM
from openfl.contracts import FLChallenge as Challenge, FLManager as Manager
from openfl.utils import require_env_var
from openfl.utils.W3Helper import get_PRIVKEYS, get_RPC_Endpoint, get_account_RPC
from openfl.utils.types.Attitude import Attitude
from types import SimpleNamespace
from web3 import Web3, Account
from openfl.api import globals

from openfl.utils.async_writer import AsyncWriter
from openfl.utils.types.User import User


def run_experiment(dataset_name: str, experiment_config: ExperimentConfiguration, writer: AsyncWriter=None, logger=None):

  dataset_name = dataset_name.replace(".", "-")
  experiment_config.dataset = dataset_name

  experiment_start = time.perf_counter()

  setup_connection(experiment_config)


  users: List[User] = []

  pytorch_model = PM.PytorchModel(
      experiment_config,
      dataset_name,
      experiment_config.number_of_good_contributors,
      experiment_config.number_of_contributors,
      experiment_config.epochs,
      experiment_config.batch_size,
      experiment_config.standard_buy_in,
      experiment_config.max_buy_in,
      experiment_config.freerider_noise_scale,
      experiment_config.freerider_start_round,
      experiment_config.malicious_start_round,
      experiment_config.malicious_noise_scale,
      experiment_config.force_merge_all)

  for attitude, count in [
      (Attitude.Honest, experiment_config.number_of_good_contributors),
      (Attitude.Malicious, experiment_config.number_of_bad_contributors),
      (Attitude.FreeRider, experiment_config.number_of_freerider_contributors),
  ]:
      for _ in range(count):
          user_index = len(users)
          addr, private_key = get_account_RPC(user_index, experiment_config.fork)
          user = User.from_experiment_config(
              attitude,
              experiment_config,
              addr,
              private_key
          )
          apply_user_data_and_label_config(user, user_index, experiment_config)
          users.append(user)

  pytorch_model.prepare_data_for_users(users, dataset_name)

  publisher: User = users[0]

  RPC_ENDPOINT = get_RPC_Endpoint()
  PRIVKEYS = get_PRIVKEYS(experiment_config) # TODO : HUH, private keys?

  manager = Manager(pytorch_model, publisher,True).init(experiment_config.number_of_good_contributors,
                                              experiment_config.number_of_bad_contributors,
                                              experiment_config.number_of_freerider_contributors,
                                              experiment_config.number_of_inactive_contributors,
                                              experiment_config.minimum_rounds,
                                              RPC_ENDPOINT,
                                              PRIVKEYS)


  training_specs = experiment_config.get_training_specs(manager.contract.address, pytorch_model.get_global_model_hash())
  
  new_job_listing: JobListing = publisher.deploy_joblisting_contract(training_specs, manager)

  writer.write_comment(f"$startingUserConfig${[p.get_status() for p in pytorch_model.participants]}")

  extra_configs = {}
  if experiment_config.contribution_score_strategy is not None:
      extra_configs["contribution_score_strategy"] = (
          experiment_config.contribution_score_strategy
      ) # WTF is this????


  for user in users:
     user.register_for_job(new_job_listing)

  while True:
      try:
          (receipt, events) = new_job_listing.transact(
              "decideOnParticpants",
              publisher,
              0,
              ["SelectionComplete"],
              "JobListing.decideOnParticpants",
              experiment_config.number_of_participants
          )
          participants_addresses = events["SelectionComplete"][0]["participants"]
          break
      except ContractLogicError as e:
          if "AWO" in str(e):
              globals.w3.provider.make_request("evm_increaseTime", [30])
              globals.w3.provider.make_request("evm_mine", [])
              print("Application window still open, trying again in 10 seconds")
              time.sleep(10)
          else:
              raise

  trainingSpecsChallenge = training_specs.to_challenge(experiment_config.contribution_score_strategy, experiment_config.use_outlier_detection, new_job_listing.contract.address)

  newChallenge: Challenge = publisher.deploy_challenge_contract(trainingSpecsChallenge, new_job_listing, pytorch_model, writer, logger)

  participating_users = get_users_from_addresses(users, participants_addresses)

  experiment_finger_print = experiment_config.get_finger_print(newChallenge)

  newChallenge.pytorch_model.setup_replay(experiment_finger_print, experiment_config)

  newChallenge.make_participants_from_users(participating_users)
  for user in newChallenge.pytorch_model.participants:
      try:
        newChallenge.transact("registrationProcess", user, trainingSpecsChallenge.min_collateral, [], "challenge.register")
      except ContractLogicError as e:
          if "SUO" in str(e):
              print("Participant tried joining but was not selected")

  # This happens after deciding on users
  newChallenge.simulate(rounds=experiment_config.minimum_rounds)
  experiment_end = time.perf_counter()
  total_experiment_time = experiment_end - experiment_start

  print("\n" + "="*75)
  print(f"TOTAL EXPERIMENT TIME: {total_experiment_time:.2f} seconds")
  writer.write_comment(f"TOTAL EXPERIMENT TIME: {total_experiment_time:.2f} seconds")
  print("="*75 + "\n")

  if logger is not None:
      try:
          import torch
          gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
      except (ImportError, Exception):
          gpu_name = "Unknown"

      hardware = {
          "cpu_name":  platform.processor(),
          "cpu_cores": psutil.cpu_count(logical=False),
          "ram_gb":    round(psutil.virtual_memory().total / (1024**3), 2),
          "gpu_name":  gpu_name,
          "os_name":   platform.system(),
      }

      cfg = experiment_config
      config = {
          "contribution_score_strategy":       cfg.contribution_score_strategy,
          "use_outlier_detection":             cfg.use_outlier_detection,
          "number_of_good_contributors":       cfg.number_of_good_contributors,
          "number_of_bad_contributors":        cfg.number_of_bad_contributors,
          "number_of_freerider_contributors":  cfg.number_of_freerider_contributors,
          "number_of_inactive_contributors":   cfg.number_of_inactive_contributors,
          "reward":                            cfg.reward,
          "minimum_rounds":                    cfg.minimum_rounds,
          "min_buy_in":                        cfg.min_buy_in,
          "max_buy_in":                        cfg.max_buy_in,
          "standard_buy_in":                   cfg.standard_buy_in,
          "epochs":                            cfg.epochs,
          "batch_size":                        cfg.batch_size,
          "punish_factor":                     cfg.punish_factor,
          "punish_factor_contrib":             cfg.punish_factor_contrib,
          "first_round_fee":                   cfg.first_round_fee,
          "fork":                              cfg.fork,
          "dataset":                           cfg.dataset,
          "freerider_start_round":             cfg.freerider_start_round,
          "freerider_noise_scale":             cfg.freerider_noise_scale,
          "malicious_start_round":             cfg.malicious_start_round,
          "malicious_noise_scale":             cfg.malicious_noise_scale,
          "force_merge_all":                   cfg.force_merge_all,
      }

      logger.log_setup(total_experiment_time, hardware, config)

  return Experiment(newChallenge, manager)


def apply_user_data_and_label_config(user: User, user_index: int, experiment_config: ExperimentConfiguration):
    user.data_percent = float(experiment_config.data_percentages[user_index])

    user_rule = experiment_config.label_rules.get(user_index, {})
    user.only_labels = user_rule.get("only_labels")
    user.flip_map = user_rule.get("flip_map", {})


def setup_connection(experiment_config):
    RPC_ENDPOINT = require_env_var("RPC_URL")

    # Only for the real-net simulation
    # In order to use a non-locally forked blockchain,
    # private keys are required to unlock accounts
    if experiment_config.fork == False:
        globals.w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))

        raw_keys = require_env_var("PRIVATE_KEYS")
        privKeys = [k.strip() for k in raw_keys.splitlines() if k.strip()]

        # Convert to Web3 Account objects
        loaded_accounts = [Account.from_key(k) for k in privKeys]

        # Wrap for compatibility with older code expecting `.privateKey`
        PRIVKEYS = [
            SimpleNamespace(privateKey=acc._private_key, address=acc.address)
            for acc in loaded_accounts
        ]

        print(f"Loaded {len(PRIVKEYS)} private keys.")
    else:
        PRIVKEYS = None

def visualizeModel(model):
  model.visualize_simulation("figures")

def get_users_from_addresses(users, addresses):
    found_users = []
    for user in users:
        for address in addresses:
            if user.address == address:
                found_users.append(user)
    return found_users

def print_transactions(experiment):
  model = experiment.model
  print("{:<10} - {:^64} -    Gas Used - {}".format("Function", "Transaction Hash", "Success"))
  print("------------------------------------------------------------------------------------------")
  for f, txhash, gasUsed in model.txHashes:
      r = globals.w3.eth.wait_for_transaction_receipt(txhash)
      if r["status"] == 1:
          success = "✅"
      else:
          success = "FAIL"
      
      gas = r["gasUsed"]
      print("{:<10} - {} - {:>9,.0f} -   {}".format(f, txhash, gas, success))


def print_latex(experiment):
  model = experiment.model
  manager = experiment.manager
  print("\\renewcommand{\\arraystretch}{1.3}")
  print("\\begin{center}")
  print("\\begin{tabular}{ c|c }")

  print("Contract & Address (Ropsten Testnet) \\\\")
  print("\\hline")
  print("Ma-1 & {} \\ ".format(manager.manager.address))
  print("Ch-1 & {} \\ ".format(model.model.address))
  for i, p in enumerate(model.pytorch_model.participants[:-1] + \
                            model.pytorch_model.disqualified + \
                            [model.pytorch_model.participants[-1]]):
      print("P-{}  & {} \\ ".format(i+1, p.address))

  print("\\end{tabular}")
  print("\\end{center}")


def table_with_gas_and_transactions_latex(experiment):
  model = experiment.model
  manager = experiment.manager
  reg = model.gas_register, "register"
  fed = model.gas_feedback, "feedback"
  clo = model.gas_close, "settle round"
  slo = model.gas_slot, "reserve slot"
  wei = model.gas_weights, "provide weights**"
  dep = manager.gas_deploy, "deployment"
  dep = manager.gas_deploy, "deployment"
  ext = model.gas_exit, "exit"

  tot  = 0
  tot2 = 0

  print("\\begin{tabular}{ |c|c|c| }\n\\hline\nFunction & Gas Amount & Gas Costs*\\\\ \n\\hline")
  for i, f in [reg,slo,wei,fed,clo]:
      print("{} & {:,.0f} & {:.5f} ETH \\\\".format(f, sum(i)/len(i), sum(i)/len(i) * 20e9 / 1e18 ))
      tot += sum(i)/len(i)
      if i != clo[0]:
              tot2 += sum(i)/len(i)
          
  print("\\hline\n\\hline")
  print("complete round & {:,.0f} & {:.5f} \\ ".format(tot, tot * 20e9 / 1e18))
  print("\\hline\n\\end{tabular}")

class Experiment:
  def __init__(self, model, manager):
    self.model = model
    self.manager = manager
