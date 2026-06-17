"""Disqualified (kicked) participants must be punished at settlement, not frozen.

Regression coverage for the fix where ``PytorchModel.get_participant`` now
resolves users that have been moved into ``self.disqualified`` (kicked mid-task).
Previously such users were dropped from the TR settlement write-back, so their
TaskRep, balance and GIR all froze at their pre-kick values instead of decaying.

Two layers are covered:
  * ``PytorchModel.get_participant`` default list includes disqualified users
    (the root fix — without it the on-chain ``taskRepDelta = -1e18`` punishment
    never reaches the Python/manager write-back).
  * ``multirep._apply_trs_reps`` applied with a disqualification entry
    ``(guid, -1e18, -1e18, pos, total)`` drives contrib score to the floor and
    decays TaskRep + balance (+ GIR in PerTask, untouched in GlobalOnly).

experiment/multirep.py is loaded directly by path because the package
experiment/multirep/ shadows the module name on a normal import (same trick as
test_apply_trs_reps_globalrep.py).
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import openfl.ml.pytorch_model as pm

_WAD = 10 ** 18
_REPO = Path(__file__).resolve().parents[2]
MNIST, CIFAR = 5, 6


@pytest.fixture(scope="module")
def mr():
    spec = importlib.util.spec_from_file_location(
        "multirep_under_test", _REPO / "experiment" / "multirep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeManager:
    """Mirrors OpenFLManager._repKey aliasing for the two modes."""

    def __init__(self, global_rep_only: bool):
        self.global_rep_only = global_rep_only
        self.task_rep: dict = {}
        self.calc: dict = {}
        self.count: dict = {}
        self.gir: dict = {}
        self.balance: dict = {}
        self.gir_calls = 0

    def _key(self, tt):
        return 0 if self.global_rep_only else tt

    def get_task_rep_calc_state(self, addr, tt):
        return self.calc.get((addr, self._key(tt)), (0, 0))

    def set_user_task_rep(self, addr, tt, v):
        self.task_rep[(addr, self._key(tt))] = v

    def set_task_rep_calc_state(self, addr, tt, mean, m2):
        self.calc[(addr, self._key(tt))] = (mean, m2)

    def increment_task_count(self, addr, tt):
        k = (addr, self._key(tt))
        self.count[k] = self.count.get(k, 0) + 1

    def set_user_integrity_rep(self, addr, v):
        self.gir_calls += 1
        self.gir[addr] = v

    def set_user_balance(self, addr, v):
        self.balance[addr] = v


class _FakeUser:
    def __init__(self, guid="g1", addr="0xabc", number=1):
        self.guid = guid
        self.address = addr
        self.number = number
        self.task_rep: dict = {}
        self.q_value: dict = {}
        self.task_count: dict = {}
        self.global_integrity_rep = 0
        self.balance = 0
        self.total_contrib_score = 0
        self.partition_spec = None


# A disqualification settlement entry: the contract sets taskRepDelta = -1e18 and
# globalReputationScore = 0, so delta_balance = 0 - 1 ETH stake = -1e18.
def _good(u, pos=5, total=5):
    return (u.guid, 2 * _WAD, _WAD, pos, total)


def _kicked(u, pos=0, total=4):
    return (u.guid, -_WAD, -_WAD, pos, total)


# ---------------------------------------------------------------------------
# Root fix: get_participant default resolves disqualified users
# ---------------------------------------------------------------------------

def test_get_participant_default_includes_disqualified():
    kicked = SimpleNamespace(id=None, address="0xdead", guid="gk")
    model = SimpleNamespace(participants=[], disqualified=[kicked])
    # Bound-method call via the class so PytorchModel.__init__ (heavy) is skipped.
    found = pm.PytorchModel.get_participant(model, "0xdead")
    assert found is kicked, "kicked user must resolve through the default list"


def test_get_participant_default_still_finds_active():
    active = SimpleNamespace(id=None, address="0xbeef", guid="ga")
    kicked = SimpleNamespace(id=None, address="0xdead", guid="gk")
    model = SimpleNamespace(participants=[active], disqualified=[kicked])
    assert pm.PytorchModel.get_participant(model, "0xbeef") is active
    assert pm.PytorchModel.get_participant(model, "0xdead") is kicked


def test_get_participant_explicit_list_overrides_default():
    a = SimpleNamespace(id=None, address="0xaaa", guid="g")
    model = SimpleNamespace(participants=[], disqualified=[])
    assert pm.PytorchModel.get_participant(model, "0xaaa", [a]) is a


# ---------------------------------------------------------------------------
# Contribution-score transform: the disqualification delta floors at zero
# ---------------------------------------------------------------------------

def test_transform_delta_disqualification_clamps_to_zero(mr):
    assert mr._transform_delta(-_WAD, _WAD, 10 * _WAD, 5) == 0


def test_transform_delta_positive_is_above_zero(mr):
    assert mr._transform_delta(2 * _WAD, _WAD, 10 * _WAD, 5) > 0


# ---------------------------------------------------------------------------
# Settlement write-back: kicked entry decays, never freezes (PerTask = multirep)
# ---------------------------------------------------------------------------

def test_pertask_kick_decays_taskrep_and_gir_and_balance(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=False)

    mr._apply_trs_reps([u], [_good(u)], MNIST, mgr, reward=10 * _WAD)
    tr_good = u.task_rep[MNIST]
    gir_good = u.global_integrity_rep
    bal_good = u.balance
    assert tr_good > 0 and gir_good > 0 and bal_good > 0

    mr._apply_trs_reps([u], [_kicked(u)], MNIST, mgr, reward=10 * _WAD)
    assert u.task_rep[MNIST] < tr_good, "TaskRep must drop on disqualification, not freeze"
    assert u.global_integrity_rep < gir_good, "GIR must decay on disqualification"
    assert u.balance < bal_good, "balance must drop on disqualification"


def test_pertask_kick_balance_floored_at_zero_on_chain(mr):
    u = _FakeUser()  # balance starts at 0
    mgr = _FakeManager(global_rep_only=False)
    mr._apply_trs_reps([u], [_kicked(u)], MNIST, mgr, reward=10 * _WAD)
    # Python balance may go negative; the manager (on-chain mirror) clamps at 0.
    assert u.balance < 0
    assert mgr.balance[u.address] == 0


# ---------------------------------------------------------------------------
# Settlement write-back: GlobalOnly (globalrep) decays TaskRep but never GIR
# ---------------------------------------------------------------------------

def test_globalonly_kick_decays_taskrep_but_not_gir(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=True)

    mr._apply_trs_reps([u], [_good(u)], MNIST, mgr, reward=10 * _WAD)
    tr_good = u.task_rep[MNIST]
    assert tr_good > 0

    mr._apply_trs_reps([u], [_kicked(u)], MNIST, mgr, reward=10 * _WAD)
    assert u.task_rep[MNIST] < tr_good, "shared TaskRep must drop on disqualification"
    assert u.global_integrity_rep == 0, "GIR must stay frozen in GlobalOnly"
    assert mgr.gir_calls == 0, "set_user_integrity_rep must not be called in GlobalOnly"
