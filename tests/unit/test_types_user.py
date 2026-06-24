"""Unit tests for the User domain object.

User.__init__ pulls collateral jitter + a secret nonce from a module-level numpy
RNG and computes a color. The existing test_user_finger_print.py covers the
finger_print hash via a __new__ bypass; here we exercise the *constructor* and
behavioural methods, patching the module RNG for determinism.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import openfl.utils.types.User as user_mod
from openfl.utils.types.Attitude import Attitude
from openfl.utils.types.User import User


class _FixedRNG:
    """Deterministic stand-in for numpy's Generator: every draw returns 5."""
    def integers(self, low, high, dtype=None):
        return 5


@pytest.fixture
def make_user(monkeypatch):
    """Factory building a fully-constructed User with the RNG pinned."""
    monkeypatch.setattr(user_mod, "RNG", _FixedRNG())

    def _make(attitude=Attitude.Honest, default_collateral=100, max_collateral=200,
              address="0x" + "a" * 40, private_key="pk", data_percent=10.0,
              only_labels=None, attitude_switch=1, number_of_participants=None):
        return User(attitude, default_collateral, max_collateral, address, private_key,
                    data_percent, only_labels, attitude_switch, number_of_participants)
    return _make


def test_init_sets_collateral_with_jitter_and_defaults(make_user):
    u = make_user(attitude=Attitude.FreeRider, default_collateral=100, max_collateral=200,
                  attitude_switch=3, data_percent=12.5, only_labels=[2, 1])

    assert u.collateral == 105            # lo(100) + fixed jitter(5)
    assert u.secret == 5
    assert u.attitude is Attitude.Honest  # everyone starts honest
    assert u.futureAttitude is Attitude.FreeRider
    assert u.attitudeSwitch == 3
    assert u.data_percent == 12.5
    assert u.only_labels == [2, 1]
    # Reputation/bookkeeping defaults.
    assert u.task_rep == {} and u.q_value == {} and u.balance == 0
    assert u.isRegistered is False


def test_init_zero_range_collateral_has_no_jitter(make_user):
    u = make_user(default_collateral=100, max_collateral=100)
    assert u.collateral == 100  # diff == 0 -> no jitter draw


def test_init_rejects_inverted_collateral_range(make_user):
    with pytest.raises(ValueError):
        make_user(default_collateral=200, max_collateral=100)


@pytest.mark.parametrize("attitude,exp_switch,exp_noise,exp_start", [
    (Attitude.Malicious, 4, 0.5, 4),
    (Attitude.FreeRider, 2, 0.1, 2),
    (Attitude.Honest, 1, None, None),
])
def test_from_experiment_config_branches(monkeypatch, attitude, exp_switch, exp_noise, exp_start):
    monkeypatch.setattr(user_mod, "RNG", _FixedRNG())
    exp = SimpleNamespace(
        min_buy_in=100, max_buy_in=200,
        malicious_start_round=4, malicious_noise_scale=0.5,
        freerider_start_round=2, freerider_noise_scale=0.1,
    )

    u = User.from_experiment_config(attitude, exp, "0x" + "b" * 40, "pk")

    assert u.futureAttitude is attitude
    assert u.attitudeSwitch == exp_switch
    assert u.noise_scale == exp_noise
    assert u.start_round == exp_start
    assert u.min_collateral == 100 and u.max_collateral == 200


def test_display_label_uses_partition_name_then_number(make_user):
    u = make_user()
    assert u.display_label() == f"#{u.number}"
    u.partition_name = "User 3"
    assert u.display_label() == "User 3"


def test_guid_follows_partition_spec(make_user):
    u = make_user()
    assert u.guid is None
    u.partition_spec = SimpleNamespace(guid="g-123")
    assert u.guid == "g-123"


def test_get_id_or_address_prefers_id(make_user):
    u = make_user()
    u.id = None
    assert u.get_id_or_address() == u.address  # falls back to address
    u.id = 7
    assert u.get_id_or_address() == 7


def test_get_id_or_address_raises_when_both_missing(make_user):
    u = make_user()
    u.id = None
    u.address = None
    with pytest.raises(ValueError):
        u.get_id_or_address()


def test_get_status_contains_identity_fields(make_user):
    u = make_user(address="0x" + "c" * 40)
    status = u.get_status()
    assert status.startswith("$user$")
    assert u.address in status


def test_reset_for_experiment_clears_registration(make_user):
    u = make_user()
    u.isRegistered = True
    u.id = 9
    u.txs = ["tx"]
    old_secret = u.secret

    u.reset_for_experiment()

    assert u.isRegistered is False
    assert u.id is None
    assert u.txs == []
    assert u.secret == old_secret  # _FixedRNG re-draws the same value


def test_to_dict_excludes_private_and_callable_fields(make_user):
    u = make_user()
    d = u.to_dict()
    assert "address" in d
    assert "private_key" not in d   # "private" substring is filtered
    assert "privateKey" not in d
    assert all(not callable(v) for v in d.values())


def test_update_color_maps_attitude(make_user):
    u = make_user()
    u.update_color(0, "bad")
    assert u.color == user_mod.get_color(0, "bad")


def test_register_for_job_transacts_and_records_txhash(make_user):
    u = make_user(address="0x" + "d" * 40)
    receipt = {"transactionHash": bytes.fromhex("ab" * 32)}
    job = MagicMock()
    job.transact.return_value = (receipt, None)

    result = u.register_for_job(job)

    job.transact.assert_called_once_with(
        "register", u, u.collateral, [], "User.register_for_job",
        bytes.fromhex(u.finger_print),
    )
    assert result == [receipt["transactionHash"]]
    assert u.txs == [receipt["transactionHash"]]


def test_batch_register_for_job_records_each_txhash(make_user):
    users = [make_user(address="0x" + "e" * 40), make_user(address="0x" + "f" * 40)]
    receipts = [{"transactionHash": bytes.fromhex("11" * 32)},
                {"transactionHash": bytes.fromhex("22" * 32)}]
    job = MagicMock()
    job.batch_transact.return_value = [(receipts[0], None), (receipts[1], None)]

    User.batch_register_for_job(users, job)

    expected_calls = [(u, u.collateral, bytes.fromhex(u.finger_print)) for u in users]
    job.batch_transact.assert_called_once_with(
        "register", expected_calls, [], "User.register_for_job",
    )
    assert users[0].txs == [receipts[0]["transactionHash"]]
    assert users[1].txs == [receipts[1]["transactionHash"]]
