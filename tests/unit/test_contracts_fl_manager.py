from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from web3 import Web3

from openfl.contracts.FLManager import FLManager, build_task_type_enum


# A valid checksum-able address used wherever FLManager calls
# Web3.to_checksum_address on its argument.
USER_ADDR = "0x" + "1" * 40


class _DummyJobListing:
    def __init__(self):
        self.contract = SimpleNamespace(address="0xJobListing")


def test_init_sets_core_fields():
    publisher = SimpleNamespace(address="0xPublisher")

    mgr = FLManager(publisher=publisher)

    assert mgr.publisher is publisher
    assert mgr.manual_setup is False
    assert mgr.latestBlock is None
    assert mgr.contract is None
    assert mgr.challenge_contract is None
    assert mgr.modelOf == {}
    assert mgr.job_listings == []
    assert mgr.gas_deploy == []
    assert mgr.txHashes == []
    assert mgr.job_template_address is None


def test_init_honours_manual_ganache_flag():
    mgr = FLManager(publisher=SimpleNamespace(), manual_ganache_setup=True)
    assert mgr.manual_setup is True


def test_get_model_of_queries_contract():
    class FakeCall:
        def __init__(self, result):
            self._result = result
            self.called_with = None

        def call(self, params):
            self.called_with = params
            return self._result

    captured = {}

    class FakeFunctions:
        def getModel(self, participant_address, addr):
            captured["args"] = (participant_address, addr)
            return FakeCall("model-result")

    fake_contract = SimpleNamespace(functions=FakeFunctions(), address="0xManager")

    mgr = FLManager(publisher=SimpleNamespace())
    mgr.contract = fake_contract

    participant = SimpleNamespace(address="0xParticipant")
    result = mgr.get_model_of(participant, "0xModelAddr")

    assert result == "model-result"
    assert captured["args"] == ("0xParticipant", "0xModelAddr")


# --------------------------------------------------------------------------- #
# build_task_type_enum (pure)
# --------------------------------------------------------------------------- #
def test_build_task_type_enum_maps_names_to_ordinals():
    TaskType = build_task_type_enum(["template", "Images", "MNIST"])
    assert TaskType.template == 0
    assert TaskType.Images == 1
    assert TaskType.MNIST == 2
    assert TaskType(1).name == "Images"


# --------------------------------------------------------------------------- #
# attach_existing / assert_reputation_mode — ReputationMode guard
# --------------------------------------------------------------------------- #
def _manager_contract_with_mode(mode):
    contract = MagicMock()
    contract.address = "0xManager"
    contract.functions.reputationMode.return_value.call.return_value = mode
    return contract


def test_attach_existing_binds_matching_per_task_manager():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"), global_rep_only=False)
    contract = _manager_contract_with_mode(FLManager.REPUTATION_MODE_PER_TASK)

    returned = mgr.attach_existing(contract)

    assert returned is mgr
    assert mgr.contract is contract


def test_attach_existing_rejects_mode_mismatch():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"), global_rep_only=False)
    contract = _manager_contract_with_mode(FLManager.REPUTATION_MODE_GLOBAL_ONLY)

    with pytest.raises(ValueError):
        mgr.attach_existing(contract)


def test_attach_existing_binds_matching_global_only_manager():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"), global_rep_only=True)
    contract = _manager_contract_with_mode(FLManager.REPUTATION_MODE_GLOBAL_ONLY)

    assert mgr.attach_existing(contract) is mgr


def test_assert_reputation_mode_passes_when_matching():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"), global_rep_only=False)
    mgr.contract = _manager_contract_with_mode(FLManager.REPUTATION_MODE_PER_TASK)
    assert mgr.assert_reputation_mode() is mgr


def test_assert_reputation_mode_raises_on_mismatch():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"), global_rep_only=True)
    mgr.contract = _manager_contract_with_mode(FLManager.REPUTATION_MODE_PER_TASK)
    with pytest.raises(ValueError):
        mgr.assert_reputation_mode()


# --------------------------------------------------------------------------- #
# register_joblisting_contract
# --------------------------------------------------------------------------- #
def _mgr_with_stubbed_transact():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"))
    mgr.transact = MagicMock()
    return mgr


def test_register_joblisting_contract_accepts_valid_listing():
    mgr = _mgr_with_stubbed_transact()
    mgr.transact.return_value = (MagicMock(), {"JobListingValid": [{"isValid": True}]})
    job = SimpleNamespace(publisher=SimpleNamespace(address="0xJobPub"),
                          contract=SimpleNamespace(address="0xJob"))

    assert mgr.register_joblisting_contract(job) is True
    assert job in mgr.job_listings
    mgr.transact.assert_called_once_with(
        "registerJob", job.publisher, 0, ["JobListingValid"], "Manager.registerJob", "0xJob",
    )


def test_register_joblisting_contract_rejects_invalid_listing():
    mgr = _mgr_with_stubbed_transact()
    mgr.transact.return_value = (MagicMock(), {"JobListingValid": [{"isValid": False}]})
    job = SimpleNamespace(publisher=SimpleNamespace(address="0xJobPub"),
                          contract=SimpleNamespace(address="0xJob"))

    assert mgr.register_joblisting_contract(job) is False
    assert job not in mgr.job_listings


# --------------------------------------------------------------------------- #
# Setter marshalling — pin the on-chain call contract (func name, gas tag,
# checksummed address, positional args). The real effect lives in Solidity.
# --------------------------------------------------------------------------- #
def test_set_user_integrity_rep_marshals_call():
    mgr = _mgr_with_stubbed_transact()
    mgr.set_user_integrity_rep(USER_ADDR, 999)
    mgr.transact.assert_called_once_with(
        "setUserIntegrityRep", mgr.publisher, 0, [], "manager.setUserIntegrityRep",
        Web3.to_checksum_address(USER_ADDR), 999,
    )


def test_set_user_task_rep_marshals_call():
    mgr = _mgr_with_stubbed_transact()
    mgr.set_user_task_rep(USER_ADDR, 6, 42)
    mgr.transact.assert_called_once_with(
        "setUserTaskRep", mgr.publisher, 0, [], "manager.setUserTaskRep",
        Web3.to_checksum_address(USER_ADDR), 6, 42,
    )


def test_increment_task_count_marshals_call():
    mgr = _mgr_with_stubbed_transact()
    mgr.increment_task_count(USER_ADDR, 6)
    mgr.transact.assert_called_once_with(
        "incrementTaskCount", mgr.publisher, 0, [], "manager.incrementTaskCount",
        Web3.to_checksum_address(USER_ADDR), 6,
    )


def test_set_task_rep_calc_state_marshals_call():
    mgr = _mgr_with_stubbed_transact()
    mgr.set_task_rep_calc_state(USER_ADDR, 6, 100, 200)
    mgr.transact.assert_called_once_with(
        "setTaskRepCalcState", mgr.publisher, 0, [], "manager.setTaskRepCalcState",
        Web3.to_checksum_address(USER_ADDR), 6, 100, 200,
    )


def test_update_q_values_after_selection_checksums_all_lists():
    mgr = _mgr_with_stubbed_transact()
    other = "0x" + "2" * 40
    mgr.update_q_values_after_selection([USER_ADDR, other], [USER_ADDR], 6, hard_reset=True)
    mgr.transact.assert_called_once_with(
        "updateQValuesAfterSelection", mgr.publisher, 0, [], "manager.updateQValuesAfterSelection",
        [Web3.to_checksum_address(USER_ADDR), Web3.to_checksum_address(other)],
        [Web3.to_checksum_address(USER_ADDR)], 6, True,
    )


def test_apply_precomputed_task_reps_marshals_records():
    mgr = _mgr_with_stubbed_transact()
    records = [("rec", 1)]
    mgr.apply_precomputed_task_reps(records, 6)
    mgr.transact.assert_called_once_with(
        "applyPrecomputedTaskReps", mgr.publisher, 0, [], "manager.applyPrecomputedTaskReps",
        records, 6,
    )


def test_initialize_user_balances_sets_each_user():
    mgr = _mgr_with_stubbed_transact()
    users = [SimpleNamespace(address=USER_ADDR), SimpleNamespace(address="0x" + "2" * 40)]
    mgr.initialize_user_balances(users, initial_value=7)

    assert mgr.transact.call_count == 2
    # Every call routes through setUserIntegrityRep with the shared initial value.
    for args in mgr.transact.call_args_list:
        assert args.args[0] == "setUserIntegrityRep"
        assert args.args[-1] == 7


def test_batch_seed_rep_state_builds_per_user_calls():
    mgr = _mgr_with_stubbed_transact()
    mgr.batch_transact = MagicMock()
    rep_state = {USER_ADDR: {"tr": 1, "gir": 2, "c_mean": 3, "m2": 4, "k": 5}}  # no "q" -> defaults 0

    mgr.batch_seed_rep_state(rep_state, task_type=6)

    mgr.batch_transact.assert_called_once()
    func_name, calls, events, gas_type = mgr.batch_transact.call_args.args
    assert func_name == "seedRepState"
    assert events == [] and gas_type == "manager.seedRepState"
    assert calls == [
        (mgr.publisher, 0, Web3.to_checksum_address(USER_ADDR), 6, 1, 2, 3, 4, 5, 0)
    ]


# --------------------------------------------------------------------------- #
# Read-only getter wrappers
# --------------------------------------------------------------------------- #
def test_get_task_type_names_returns_list_from_contract():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"))
    mgr.contract = MagicMock()
    mgr.contract.functions.getTaskTypeNames.return_value.call.return_value = ["template", "Images", "MNIST"]

    assert mgr.get_task_type_names() == ["template", "Images", "MNIST"]


def test_get_task_type_enum_builds_from_contract_names():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"))
    mgr.contract = MagicMock()
    mgr.contract.functions.getTaskTypeNames.return_value.call.return_value = ["template", "Images", "MNIST"]

    TaskType = mgr.get_task_type_enum()
    assert TaskType.MNIST == 2


def test_get_task_rep_calc_state_checksums_and_returns_call():
    mgr = FLManager(publisher=SimpleNamespace(address="0xPub"))
    mgr.contract = MagicMock()
    mgr.contract.functions.getTaskRepCalcState.return_value.call.return_value = (10, 20)

    assert mgr.get_task_rep_calc_state(USER_ADDR, 6) == (10, 20)
    mgr.contract.functions.getTaskRepCalcState.assert_called_once_with(
        Web3.to_checksum_address(USER_ADDR), 6,
    )
