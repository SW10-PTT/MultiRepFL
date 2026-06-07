"""Tests for the GlobalOnly fix in multirep._apply_trs_reps.

Verifies the replay write-back honours global_rep_only the same way the
OpenFLManager contract does (_repKey + applyGIR=false):
  * GlobalOnly keeps GIR at its prior value (no integrity updates).
  * GlobalOnly aliases TaskRep onto one shared bucket that compounds across
    task types instead of resetting on each dataset switch.
  * PerTask keeps per-task-type TaskRep and updates GIR from votes.

experiment/multirep.py is loaded directly by path because the package
experiment/multirep/ shadows the module name on a normal import.
"""

import importlib.util
from pathlib import Path

import pytest

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


def _apply(mr, users, mgr, task_type, delta=2 * _WAD, pos=5, total=5, reward=10 * _WAD):
    trs = [(u.guid, delta, _WAD, pos, total) for u in users]
    return mr._apply_trs_reps(users, trs, task_type, mgr, reward)


# ---------------------------------------------------------------------------
# GlobalOnly
# ---------------------------------------------------------------------------

def test_globalonly_gir_stays_zero_with_perfect_votes(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=True)
    _apply(mr, [u], mgr, MNIST, pos=5, total=5)
    assert u.global_integrity_rep == 0, "GIR must not move in GlobalOnly"
    assert mgr.gir_calls == 0, "set_user_integrity_rep must not be called in GlobalOnly"


def test_globalonly_taskrep_single_bucket_aliased(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=True)
    _apply(mr, [u], mgr, MNIST)
    # the single bucket is mirrored onto every real task-type key
    assert u.task_rep[MNIST] == u.task_rep[CIFAR] > 0


def test_globalonly_taskrep_compounds_across_task_types(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=True)
    _apply(mr, [u], mgr, MNIST)
    after_mnist = u.task_rep[MNIST]
    _apply(mr, [u], mgr, CIFAR)          # different dataset
    after_cifar = u.task_rep[CIFAR]
    assert after_cifar > after_mnist, "TR must compound, not reset, on dataset switch"
    assert u.task_rep[MNIST] == u.task_rep[CIFAR], "both keys stay aliased to the bucket"


# ---------------------------------------------------------------------------
# PerTask (multi-rep) — unchanged behaviour
# ---------------------------------------------------------------------------

def test_pertask_gir_increases_with_positive_votes(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=False)
    _apply(mr, [u], mgr, MNIST, pos=5, total=5)
    assert u.global_integrity_rep > 0
    assert mgr.gir_calls == 1


def test_pertask_taskrep_is_per_type(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=False)
    _apply(mr, [u], mgr, MNIST)
    assert u.task_rep.get(MNIST, 0) > 0
    assert u.task_rep.get(CIFAR, 0) == 0, "an MNIST task must not raise CIFAR TR in PerTask"


def test_pertask_taskrep_does_not_alias_across_types(mr):
    u = _FakeUser()
    mgr = _FakeManager(global_rep_only=False)
    _apply(mr, [u], mgr, MNIST)
    _apply(mr, [u], mgr, CIFAR)
    # independent buckets → both positive but tracked separately
    assert u.task_rep[MNIST] > 0 and u.task_rep[CIFAR] > 0
    assert (u.address, MNIST) in mgr.task_rep and (u.address, CIFAR) in mgr.task_rep
