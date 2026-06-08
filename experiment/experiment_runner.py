import hashlib
import os
import platform
from xml.dom import NotFoundErr
import psutil
import time
from pathlib import Path

from web3.exceptions import ContractLogicError
from typing import List
from experiment.experiment_configuration import ExperimentConfiguration
from openfl.contracts.JobListing import JobListing
from openfl.ml import pytorch_model as PM
from openfl.contracts import FLChallenge as Challenge, FLManager as Manager
from openfl.utils import require_env_var
from openfl.utils.ITestAndTrainer import get_filename
from openfl.utils.W3Helper import get_PRIVKEYS, get_RPC_Endpoint, get_account_RPC
from openfl.utils.types.Attitude import Attitude
from types import SimpleNamespace
from web3 import Web3, Account
from openfl.api import globals

import uuid

from openfl.utils.async_writer import AsyncWriter
from openfl.utils.types.User import User
from openfl.ml.partition_spec import UserPartitionSpec
from openfl.utils.printer import set_enabled_tags, log


class _SelectionState:
  """Carries all state from select_participants_for_task into run_experiment_from_selection."""
  __slots__ = (
      "selected_users", "all_users", "pytorch_model", "manager",
      "job_listing", "training_specs_challenge", "publisher",
      "dataset_name", "experiment_start",
  )

  def __init__(self, selected_users, all_users, pytorch_model, manager,
               job_listing, training_specs_challenge, publisher,
               dataset_name, experiment_start):
      self.selected_users = selected_users
      self.all_users = all_users
      self.pytorch_model = pytorch_model
      self.manager = manager
      self.job_listing = job_listing
      self.training_specs_challenge = training_specs_challenge
      self.publisher = publisher
      self.dataset_name = dataset_name
      self.experiment_start = experiment_start


def build_users(experiment_config: ExperimentConfiguration) -> List[User]:
  """Create User objects from an ExperimentConfiguration. Does not load data."""
  users: List[User] = []
  if experiment_config.partition_strategy == "per_user":
      # Spec drives both id→behavior mapping and data shares. Iterate spec
      # keys in sorted order so account allocation is deterministic across
      # runs regardless of JSON insertion order. Spec keys are opaque strings
      # (GUIDs or numeric strings) — the position in the sorted enumeration
      # is the on-chain account slot. Inactive entries are counted in
      # number_of_inactive_contributors but don't materialise as User
      # objects (they never join the FL round); their slot stays unused so
      # later users keep stable account assignments.
      specs = experiment_config.get_partition_specs(experiment_config.dataset)
      for account_slot, user_id in enumerate(sorted(specs.keys())):
          spec = specs[user_id]
          if spec.behavior == Attitude.Inactive:
              continue
          addr, private_key = get_account_RPC(account_slot, experiment_config.fork)
          user = User.from_experiment_config(
              spec.behavior,
              experiment_config,
              addr,
              private_key
          )
          apply_user_data_and_label_config(user, user_id, experiment_config)
          users.append(user)
  else:
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
  return users


def _log_chain_applicant_scores(users: List[User], job_listing, experiment_config) -> None:
  """Log the rep values the contract read at registration time for debugging selection divergence."""
  _WAD = 10 ** 18
  try:
      tr_w  = getattr(experiment_config, "tr_weight",  6)
      gir_w = getattr(experiment_config, "gir_weight", 4)
      q_w   = getattr(experiment_config, "q_weight",   0)
      cap_on  = getattr(experiment_config, "q_slot_limit_enabled", False)
      q_slots = getattr(experiment_config, "q_slot_limit", 0)
      denom = tr_w + gir_w
      cap_note = (f", q_slot_limit ON (limit={q_slots}: {q_slots} slots may use Q, "
                  f"rest by base score)" if cap_on else "")
      log("replay", f"  Chain applicant scores (tr={tr_w}, gir={gir_w}, q={q_w/_WAD:.4f}{cap_note}):")
      # Base = score with no Q bonus; Score = with Q. When the cap is on, base
      # decides the rep slots and Score only matters for the Q slots.
      log("replay", f"    {'Name':<16} {'TR':>8} {'GIR':>8} {'Q':>8} {'Base':>10} {'Score':>10}  {'sel':>3}  fp[:8]")
      for u in users:
          try:
              chain = job_listing.contract.functions.applicants(u.address).call()
              # struct: (globalTaskRep, globalIntegrity, qValue, tiebreaker, addr, isSelected)
              tr, gir, q_val = chain[0], chain[1], chain[2]
              is_selected = chain[5]
              base = (tr * tr_w + gir * gir_w) // denom
              q_bonus = (q_w * q_val) // _WAD
              score = base + q_bonus
              tb_hex = chain[3].hex()[:8] if isinstance(chain[3], (bytes, bytearray)) else str(chain[3])[:8]
              name = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else f"User {u.number}"
              marker = "YES" if is_selected else "no"
              log("replay", f"    {name:<16} {tr/_WAD:>8.4f} {gir/_WAD:>8.4f} {q_val/_WAD:>8.4f} {base/_WAD:>10.4f} {score/_WAD:>10.4f}  {marker:>3}  {tb_hex}")
          except Exception:
              pass
  except Exception:
      pass


def _seed_manager_rep_state(manager, rep_state: dict, task_type: int) -> None:
  """Apply prior-session rep state to a freshly deployed manager.

  rep_state maps address → {tr, gir, k, c_mean, m2} (all WAD ints).
  Must be called after manager.init() but before job listing deploy so that
  applicant scores are read with the correct TR/GIR values.
  """
  _WAD = 1e18
  manager.batch_seed_rep_state(rep_state, task_type)
  for addr, s in rep_state.items():
      log("setup", f"[rep_state] seeded {addr[:10]}... tr={s['tr']/_WAD:.4f} gir={s['gir']/_WAD:.4f} q={s.get('q',0)/_WAD:.4f} k={s['k']} task_type={task_type}")
  log("setup", f"[rep_state] seeded {len(rep_state)} users into fresh manager")


def select_participants_for_task(
    dataset_name: str,
    exp_config: ExperimentConfiguration,
    prebuilt_users: List[User] = None,
    prebuilt_manager=None,
    initial_rep_state: dict = None,
    force_remote: bool = False,
) -> _SelectionState:
  """Deploy job listing, register all users, run contract selection.

  Returns a _SelectionState with the actual chain-selected users and all
  state needed for run_experiment_from_selection. Data loading is deferred
  so the caller can check the fingerprint cache before committing to training.

  When force_remote=True, PytorchModel is not created (no GPU work). A dummy
  model hash (32 zero bytes) is posted to the on-chain JobListing instead.
  """
  set_enabled_tags(exp_config.enabled_prints)
  dataset_name = dataset_name.replace(".", "-")
  exp_config.refresh_for_dataset(dataset_name)
  experiment_start = time.perf_counter()
  setup_connection(exp_config)

  if force_remote:
      pytorch_model = None
      model_hash = b'\x00' * 32
  else:
      pytorch_model = PM.PytorchModel(
          exp_config,
          dataset_name,
          exp_config.number_of_good_contributors,
          exp_config.number_of_contributors,
          exp_config.epochs,
          exp_config.batch_size,
          exp_config.standard_buy_in,
          exp_config.max_buy_in,
          exp_config.freerider_noise_scale,
          exp_config.freerider_start_round,
          exp_config.malicious_start_round,
          exp_config.malicious_noise_scale,
          exp_config.force_merge_all,
      )
      model_hash = pytorch_model.get_global_model_hash()

  if prebuilt_users is None:
      users = build_users(exp_config)
  else:
      users = prebuilt_users
      for user in users:
          user.reset_for_experiment()

  publisher: User = users[0]

  for u in users:
      label = u.partition_spec.name if (u.partition_spec and u.partition_spec.name) else f"User {u.number}"
      globals.fp_user_labels[u.finger_print] = label

  RPC_ENDPOINT = get_RPC_Endpoint()
  PRIVKEYS = get_PRIVKEYS(exp_config)

  if prebuilt_manager is None:
      manager = Manager(publisher, True).init(
          exp_config.number_of_good_contributors,
          exp_config.number_of_bad_contributors,
          exp_config.number_of_freerider_contributors,
          exp_config.number_of_inactive_contributors,
          exp_config.minimum_rounds,
          RPC_ENDPOINT,
          PRIVKEYS,
      )
  else:
      manager = prebuilt_manager
  if pytorch_model is not None:
      manager.pytorch_model = pytorch_model

  if initial_rep_state and prebuilt_manager is None:
      from openfl.utils.types.TrainingSpecsJobListing import TaskType as _TaskType
      _task_type = int(_TaskType.from_dataset_name(dataset_name))
      _seed_manager_rep_state(manager, initial_rep_state, _task_type)

  training_specs = exp_config.get_training_specs(manager.contract.address, model_hash)
  new_job_listing: JobListing = publisher.deploy_joblisting_contract(training_specs, manager)

  User.batch_register_for_job(users, new_job_listing)

  while True:
      try:
          (_, events) = new_job_listing.transact(
              "decideOnParticpants",
              publisher,
              0,
              ["SelectionComplete"],
              "JobListing.decideOnParticpants",
              exp_config.number_of_participants,
          )
          participants_addresses = events["SelectionComplete"][0]["participants"]
          break
      except ContractLogicError as e:
          if "AWO" in str(e):
              globals.w3.provider.make_request("evm_increaseTime", [30])
              globals.w3.provider.make_request("evm_mine", [])
              log("round_boundary", "Application window still open, trying again in 10 seconds")
              time.sleep(10)
          else:
              raise

  _log_chain_applicant_scores(users, new_job_listing, exp_config)

  selected_users = get_users_from_addresses(users, participants_addresses)
  training_specs_challenge = training_specs.to_challenge(
      exp_config.contribution_score_strategy,
      exp_config.use_outlier_detection,
      new_job_listing.contract.address,
      exp_config.loss_tolerance_pct,
  )

  return _SelectionState(
      selected_users=selected_users,
      all_users=users,
      pytorch_model=pytorch_model,
      manager=manager,
      job_listing=new_job_listing,
      training_specs_challenge=training_specs_challenge,
      publisher=publisher,
      dataset_name=dataset_name,
      experiment_start=experiment_start,
  )


def run_experiment_from_selection(
    state: _SelectionState,
    exp_config: ExperimentConfiguration,
    fingerprint: str,
    writer: AsyncWriter = None,
    logger=None,
    path=None,
):
  """Run training or replay given a pre-computed participant selection.

  exp_config must be built from the actual selected users' partition specs.
  fingerprint must be exp_config.get_finger_print(state.selected_users).
  """
  if state.pytorch_model is None:
      raise RuntimeError(
          "pytorch_model is None in run_experiment_from_selection — "
          "select_participants_for_task was called with force_remote=True. "
          "Use the direct TRS extraction path in _run_remote instead."
      )
  set_enabled_tags(exp_config.enabled_prints)
  exp_config.refresh_for_dataset(state.dataset_name)

  pytorch_model = state.pytorch_model
  pytorch_model.config = exp_config
  users = state.all_users
  selected_users = state.selected_users
  manager = state.manager
  new_job_listing = state.job_listing
  training_specs_challenge = state.training_specs_challenge
  publisher = state.publisher
  dataset_name = state.dataset_name
  experiment_start = state.experiment_start

  # Load data before setup_replay: setup_replay calls runRepo.test which needs _test_data.
  # Skip in HardPlayBack mode — if no replay file is found, we load below as fallback.
  if not (globals.reuse_runs & globals.ReplayMode.HardPlayBack):
      pytorch_model.prepare_data_for_users(
          users,
          dataset_name,
          seed=exp_config.seed,
          allow_overlap=exp_config.allow_overlap,
          replication_factor=exp_config.replication_factor,
      )

  filename = get_filename(fingerprint, exp_config)
  pytorch_model.setup_replay(filename, exp_config, path)

  # HardPlayBack set but no matching file found → fall back to local training, load data now.
  if (globals.reuse_runs & globals.ReplayMode.HardPlayBack) and not (globals.reuse_runs & globals.ReplayMode._actively_replaying):
      pytorch_model.prepare_data_for_users(
          users,
          dataset_name,
          seed=exp_config.seed,
          allow_overlap=exp_config.allow_overlap,
          replication_factor=exp_config.replication_factor,
      )

  ############
  ## REPLAY ##
  ############
  flags = globals.ReplayMode._actively_replaying | globals.ReplayMode.HardPlayBack
  if (globals.reuse_runs & flags) == flags:
      log("round_models", "Replaying!")
      users_by_address = {u.address: u for u in users}
      users_list = list(users_by_address.values())
      combinedUsers = pytorch_model.runRepo.get_participants(users_list)
      trs = pytorch_model.runRepo.get_task_rep_delta_and_GRS(
          -1, "get_task_rep_delta_and_GRS-simulate", None,
          lambda x: pytorch_model.get_participant(x, combinedUsers),
      )
      pytorch_model.cleanup()
      return (trs, filename)

  newChallenge: Challenge = publisher.deploy_challenge_contract(
      training_specs_challenge, new_job_listing, pytorch_model, writer, logger, manager_contract=manager
  )

  newChallenge.make_participants_from_users(selected_users)
  for user in newChallenge.pytorch_model.participants:
      try:
          newChallenge.transact("registrationProcess", user, training_specs_challenge.min_collateral, [], "challenge.register")
      except ContractLogicError as e:
          if "SUO" in str(e):
              log("round_models", "Participant tried joining but was not selected")
  if writer is not None:
      writer.write_comment(f"$startingUserConfig${[p.get_status() for p in pytorch_model.participants]}")

  newChallenge.simulate(rounds=exp_config.minimum_rounds)

  try:
      grs_snapshot = newChallenge.contract.functions.getTaskRepDeltaAndGRS().call()
  except Exception as e:
      log("experiment_end", f"[warn] GRS snapshot failed: {e}")
      grs_snapshot = []

  # Snapshot the per-participant TaskRepRecord[] (written by computeAndRecordTaskReps
  # at the end of simulate()). Carries the transformed contribution score (cs) so
  # callers — e.g. multirep's session summary — can read the exact on-chain value.
  try:
      task_rep_records = newChallenge.contract.functions.getTaskRepRecords().call()
  except Exception as e:
      log("experiment_end", f"[warn] TaskRepRecords snapshot failed: {e}")
      task_rep_records = []

  newChallenge.exit_system()

  try:
      task_type = new_job_listing.get_task_type()
      addrs = [u.address for u in selected_users]
      reps = manager.contract.functions.getGrsAndTrsBatch(addrs, task_type).call()
      nonzero_tr = [(r[0][:10], r[1]) for r in reps if r[1] > 0]
      log("experiment_end", f"[diag] task_type={task_type} non-zero TR: {nonzero_tr or 'NONE'}")
      if selected_users:
          u = selected_users[0]
          tr, gir, _ = manager.contract.functions.getUserRep(u.address, task_type).call()
          k = manager.contract.functions.getTaskCount(u.address, task_type).call()
          cm, m2 = manager.contract.functions.getTaskRepCalcState(u.address, task_type).call()
          log("experiment_end", f"[diag] user[0] {u.address[:10]} TR={tr} GIR={gir} k={k} cMean={cm} m2={m2}")
  except Exception as e:
      log("experiment_end", f"[diag] diagnostic read failed: {e}")

  experiment_end = time.perf_counter()
  total_experiment_time = experiment_end - experiment_start

  log("experiment_end", "\n" + "="*75)
  log("experiment_end", f"TOTAL EXPERIMENT TIME: {total_experiment_time:.2f} seconds")
  if writer is not None:
      writer.write_comment(f"TOTAL EXPERIMENT TIME: {total_experiment_time:.2f} seconds")
  log("experiment_end", "="*75 + "\n")

  if logger is not None:
      _log_task_rep_calc(logger, newChallenge, manager, new_job_listing, pytorch_model,
                         grs_snapshot=grs_snapshot, task_rep_records=task_rep_records)

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

      cfg = exp_config
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
          "seed":                              cfg.seed,
          "allow_overlap":                     cfg.allow_overlap,
          "replication_factor":                cfg.replication_factor,
          "global_rep_only":                   cfg.global_rep_only,
          "user_seeds":                        {u.number: u.seed for u in users},
          "data_percentages":                  {u.number: u.data_percent for u in users},
      }

      logger.log_setup(total_experiment_time, hardware, config)

  pytorch_model.cleanup()
  return (Experiment(newChallenge, manager, grs_snapshot, task_rep_records), filename)


def run_experiment(
    dataset_name: str,
    experiment_config: ExperimentConfiguration,
    writer: AsyncWriter = None,
    logger=None,
    path=None,
    prebuilt_users: List[User] = None,
    prebuilt_manager=None,
    initial_rep_state: dict = None,
):
  """Thin wrapper: select participants via the contract, then train/replay."""
  state = select_participants_for_task(
      dataset_name, experiment_config, prebuilt_users, prebuilt_manager,
      initial_rep_state=initial_rep_state,
  )
  fingerprint = experiment_config.get_finger_print(state.selected_users)
  return run_experiment_from_selection(state, experiment_config, fingerprint, writer, logger, path)


# Run a sequence of experiments that share ONE on-chain OpenFLManager so
# reputation (TaskRep / GIR / per-user task counters) accumulates across them.
# The first run deploys the manager; every later run attaches to that same
# contract instead of redeploying, which is what makes the per-task TaskRep
# EWMA actually compound (k = 1, 2, 3, ... per user instead of resetting to 1).
#
# `jobs` is a list of (dataset_name, experiment_config) pairs, run in order.
# `make_io` is an optional callable (dataset, config) -> (writer, logger, path)
# invoked per job; return (None, None, None) to skip writer/logger wiring.
#
# Requirements / caveats for accumulation to be meaningful:
#   - Every config must agree on global_rep_only (mode is fixed on the shared
#     manager at first deploy; attach_existing raises on mismatch).
#   - The participant roster must map the same identity to the same on-chain
#     address across jobs (in per_user mode: keep the same user_index set so
#     sorted-key account slots stay stable). Differing rosters will accrue rep
#     to whoever lands on each address slot.
#   - Replaying runs (HardPlayBack) is unsupported here — run_experiment
#     returns early without an Experiment, so there is no manager to thread.
def run_experiment_sequence(jobs, make_io=None):
    results = []
    shared_manager_contract = None

    for dataset_name, experiment_config in jobs:
        writer = logger = path = None
        if make_io is not None:
            writer, logger, path = make_io(dataset_name, experiment_config)

        outcome = run_experiment(
            dataset_name,
            experiment_config,
            writer,
            logger,
            path,
            shared_manager_contract=shared_manager_contract,
        )

        if writer is not None:
            writer.finish()
        if logger is not None and path is not None:
            logger.save(path.with_suffix(".pkl"))

        # Replay path returns a non-Experiment payload (no manager to reuse);
        # surface it and stop threading rather than guessing.
        if not isinstance(outcome, tuple) or not isinstance(outcome[0], Experiment):
            results.append(outcome)
            continue

        experiment, filename = outcome
        results.append((experiment, filename))

        if shared_manager_contract is None:
            shared_manager_contract = experiment.manager.contract

    return results


def apply_user_data_and_label_config(user: User, user_index, experiment_config: ExperimentConfiguration):
    # Per-user strategy: spec drives data_percent, only_labels, flip_map,
    # behavior, noise_scale, start_round. user_index is the spec key (str).
    # Global strategy: legacy data_percentages + label_rules drive data/labels;
    # user_index is the positional int allocated by the runner; noise_scale
    # and start_round were already set by User.from_experiment_config.
    if experiment_config.partition_strategy == "per_user":
        specs = experiment_config.get_partition_specs(experiment_config.dataset)
        spec = specs[user_index]
        user.partition_spec = spec
        user.partition_name = spec.name
        user.data_percent = float(spec.data_percent)
        user.only_labels = list(spec.only_labels) if spec.only_labels is not None else None
        user.flip_map = dict(spec.flip_map)
        user.noise_scale = (
            None if spec.noise_scale is None else float(spec.noise_scale)
        )
        user.start_round = (
            None if spec.start_round is None else int(spec.start_round)
        )
        # Keep attitudeSwitch in sync with spec.start_round so the existing
        # round-gating logic in PytorchModel.update_users_attitude works.
        if spec.start_round is not None:
            user.attitudeSwitch = int(spec.start_round)
    else:
        user.data_percent = float(experiment_config.data_percentages[user_index])
        user_rule = experiment_config.label_rules.get(user_index, {})
        user.only_labels = user_rule.get("only_labels")
        user.flip_map = user_rule.get("flip_map", {})

    user.seed = derive_user_seed(experiment_config, user_index)

    if user.partition_spec is None:
        guid = str(uuid.UUID(bytes=hashlib.sha256(
            f"{experiment_config.seed}:guid:{user_index}".encode()
        ).digest()[:16]))
        noise_scale = user.noise_scale
        start_round = user.start_round
        user.partition_spec = UserPartitionSpec(
            user_index=str(user_index),
            data_percent=user.data_percent,
            only_labels=list(user.only_labels) if user.only_labels is not None else None,
            flip_map=dict(user.flip_map),
            behavior=user.futureAttitude,
            noise_scale=float(noise_scale) if noise_scale is not None else None,
            start_round=int(start_round) if start_round is not None else None,
            guid=guid,
        )


# Independent per-user RNG stream. Hashing master+user_id keeps streams
# uncorrelated and stable when users are added/removed (unlike `master+i`).
# Explicit overrides in experiment_config.user_seeds win for debug runs.
# user_index can be an int (global mode positional index) or a str (per_user
# GUID/string id); both flow through str() for a uniform lookup + payload.
def derive_user_seed(experiment_config: ExperimentConfiguration, user_index) -> int:
    key = str(user_index)
    if key in experiment_config.user_seeds:
        return int(experiment_config.user_seeds[key])
    payload = f"{experiment_config.seed}:{key}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


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

        log("setup_env", f"Loaded {len(PRIVKEYS)} private keys.")
    else:
        PRIVKEYS = None

def visualizeModel(model):
  model.visualize_simulation("figures")

def get_users_from_addresses(users, addresses):
    users_by_address = {u.address: u for u in users}

    try:
        return [users_by_address[address] for address in addresses]
    except KeyError as e:
        raise NotFoundErr(f"Address {e.args[0]} not found")

def print_transactions(experiment):
  model = experiment.model
  log("gas_report", "{:<10} - {:^64} -    Gas Used - {}".format("Function", "Transaction Hash", "Success"))
  log("gas_report", "------------------------------------------------------------------------------------------")
  for f, txhash, gasUsed in model.txHashes:
      r = globals.w3.eth.wait_for_transaction_receipt(txhash)
      if r["status"] == 1:
          success = "✅"
      else:
          success = "FAIL"

      gas = r["gasUsed"]
      log("gas_report", "{:<10} - {} - {:>9,.0f} -   {}".format(f, txhash, gas, success))


def print_latex(experiment):
  model = experiment.model
  manager = experiment.manager
  log("latex_output", "\\renewcommand{\\arraystretch}{1.3}")
  log("latex_output", "\\begin{center}")
  log("latex_output", "\\begin{tabular}{ c|c }")

  log("latex_output", "Contract & Address (Ropsten Testnet) \\\\")
  log("latex_output", "\\hline")
  log("latex_output", "Ma-1 & {} \\ ".format(manager.manager.address))
  log("latex_output", "Ch-1 & {} \\ ".format(model.model.address))
  for i, p in enumerate(model.pytorch_model.participants[:-1] + \
                            model.pytorch_model.disqualified + \
                            [model.pytorch_model.participants[-1]]):
      label = p.display_label() if hasattr(p, "display_label") else ""
      log("latex_output", "P-{} ({})  & {} \\ ".format(i+1, label, p.address))

  log("latex_output", "\\end{tabular}")
  log("latex_output", "\\end{center}")


def table_with_gas_and_transactions_latex(experiment):
  model = experiment.model
  manager = experiment.manager
  reg = model.gas_register, "register"
  fed = model.gas_feedback, "feedback"
  clo = model.gas_close, "settle round"
  slo = model.gas_slot, "reserve slot"
  wei = model.gas_weights, "provide weights**"
  con = model.gas_contrib, "contribution score"
  dep = manager.gas_deploy, "deployment"
  dep = manager.gas_deploy, "deployment"
  ext = model.gas_exit, "exit"

  tot  = 0
  tot2 = 0

  log("latex_output", "\\begin{tabular}{ |c|c|c| }\n\\hline\nFunction & Gas Amount & Gas Costs*\\\\ \n\\hline")
  for i, f in [reg,slo,wei,fed,con,clo]:
      log("latex_output", "{} & {:,.0f} & {:.5f} ETH \\\\".format(f, sum(i)/len(i), sum(i)/len(i) * 20e9 / 1e18))
      tot += sum(i)/len(i)
      if i != clo[0]:
              tot2 += sum(i)/len(i)

  log("latex_output", "\\hline\n\\hline")
  log("latex_output", "complete round & {:,.0f} & {:.5f} \\ ".format(tot, tot * 20e9 / 1e18))
  log("latex_output", "\\hline\n\\end{tabular}")

def _log_task_rep_calc(logger, challenge, manager, job_listing, pytorch_model,
                       grs_snapshot=None, task_rep_records=None):
    """Read per-participant TaskRepCalc state from the manager contract after
    updateUserTaskReps has fired and log it as the task_rep_calc table.

    Columns with WAD-normalised values (running_c_mean, m2, global_task_rep)
    are stored as floats in [0, 1]. All other on-chain integers are stored raw.

    Also prints, per participant, the transformed contribution score (cs) and the
    resulting TaskRep (TR) for this task under the task_rep_contrib tag.
    """
    WAD = 10 ** 18
    ZERO_ADDR = "0x0000000000000000000000000000000000000000"
    task_type = job_listing.get_task_type()

    # Use pre-exit snapshot when available; fall back to live contract query.
    trs_raw = grs_snapshot if grs_snapshot else challenge.contract.functions.getTaskRepDeltaAndGRS().call()
    trs_by_addr = {
        entry[0].lower(): (entry[1], entry[2], entry[3], entry[4])
        for entry in trs_raw
        if entry[0] != ZERO_ADDR
    }

    # Transformed contribution score (cs) per participant, exactly as computed
    # on-chain by _trTransformDelta inside computeAndRecordTaskReps(). We read it
    # back from the stored TaskRepRecord[] rather than re-deriving the fixed-point
    # math in Python, so the value printed is the contract's own. contribScore is
    # the last record field (index 6), WAD-scaled to [0, 1]. Prefer the pre-exit
    # snapshot; fall back to a live read (records are immutable once written).
    cs_by_addr = {}
    try:
        records = task_rep_records if task_rep_records else challenge.contract.functions.getTaskRepRecords().call()
        for rec in records:
            if rec[0] != ZERO_ADDR:
                cs_by_addr[rec[0].lower()] = rec[6]
    except Exception as e:
        log("task_rep_contrib", f"[warn] getTaskRepRecords failed, contribScore unavailable: {e}")

    all_participants = pytorch_model.participants + pytorch_model.disqualified

    log("task_rep_contrib", "\n" + "=" * 75)
    log("task_rep_contrib",
        f"CONTRIBUTION SCORE (cs = _trTransformDelta(taskRepDelta)) & RESULTING TASKREP (TR)  task_type={task_type}")
    log("task_rep_contrib", "-" * 75)

    for user in all_participants:
        addr = user.address
        e, f = manager.contract.functions.getTaskRepCalcState(addr, task_type).call()
        task_rep_wad, integrity_rep, nr_tasks = manager.contract.functions.getUserRep(addr, task_type).call()
        delta, grs, positive_votes, total_votes = trs_by_addr.get(
            addr.lower(), (None, None, None, None)
        )
        cs_wad = cs_by_addr.get(addr.lower())
        contrib_score = cs_wad / WAD if cs_wad is not None else None

        cs_str = f"{contrib_score:8.6f}" if contrib_score is not None else "     n/a"
        delta_str = f"{delta / WAD:+9.6f}" if delta is not None else "      n/a"
        tr_str = f"{task_rep_wad / WAD:8.6f}"
        log("task_rep_contrib",
            f"  user {str(user.id):>3}  {addr[:10]}...  taskRepDelta={delta_str}  ->  cs={cs_str}  TR={tr_str}")

        logger.log_task_rep_calc(
            address=addr,
            user_id=str(user.id),
            task_type=task_type,
            k=nr_tasks,
            running_c_mean=e / WAD,
            m2=f / WAD,
            global_task_rep=task_rep_wad / WAD,
            global_integrity_rep=integrity_rep / WAD,
            task_rep_delta=delta,
            final_grs=grs,
            positive_votes=positive_votes,
            total_votes=total_votes,
            contrib_score=contrib_score,
        )

    log("task_rep_contrib", "=" * 75 + "\n")


class Experiment:
  def __init__(self, model, manager, grs_snapshot=None, task_rep_records=None):
    self.model = model
    self.manager = manager
    self.grs_snapshot = grs_snapshot or []
    # Per-participant TaskRepRecord[] tuples (incl. contribScore at index 6),
    # snapshotted right after the challenge settled. [] when unavailable.
    self.task_rep_records = task_rep_records or []
