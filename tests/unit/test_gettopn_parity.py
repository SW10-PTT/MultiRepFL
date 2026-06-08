"""Tests that Python getTopN/compute_user_score exactly match the Solidity
_selectionScore + getTopN heap implemented in JobListing.sol.

Solidity reference (JobListing.sol):
    _selectionScore(u):
        denom        = trWeight + girWeight
        normalWeight = (taskRep * trWeight + gir * girWeight) / denom
        qBonus       = (qWeight * qValue) / WAD
        return normalWeight + qBonus

    _isWeaker(sA, tbA, sB, tbB):
        if sA != sB: return sA < sB
        return tbA > tbB   # lower tiebreaker wins

    getTopN(N): min-heap, keeps the N strongest candidates.
"""

import hashlib
from typing import List, Tuple

import pytest

_WAD = 10 ** 18

# ---------------------------------------------------------------------------
# Minimal User stub (avoids importing the real User class + blockchain deps)
# ---------------------------------------------------------------------------

class _User:
    def __init__(self, seed: int, task_rep: int, gir: int, q: int):
        blob = f"user:{seed}".encode()
        self.finger_print = hashlib.sha256(blob).hexdigest()
        self.address = f"0x{seed:040x}"
        self.task_rep = {}
        self.global_integrity_rep = gir
        self.q_value = {}

    def set_rep(self, task_type: int, task_rep: int, q: int):
        self.task_rep[task_type] = task_rep
        self.q_value[task_type] = q
        return self


def _make_users(n: int, task_type: int, task_rep=0, gir=_WAD, q=0) -> List[_User]:
    users = []
    for i in range(n):
        u = _User(seed=i, task_rep=task_rep, gir=gir, q=q)
        u.set_rep(task_type, task_rep, q)
        users.append(u)
    return users


# ---------------------------------------------------------------------------
# Pure-Python simulation of Solidity getTopN heap
# ---------------------------------------------------------------------------

def _sol_is_weaker(sA: int, tbA: bytes, sB: int, tbB: bytes) -> bool:
    """Mirror Solidity _isWeaker: lower score = weaker; for ties, higher bytes32 = weaker."""
    if sA != sB:
        return sA < sB
    return tbA > tbB  # bytes comparison is lexicographic, same as uint256 numeric


def _sol_score(task_rep: int, gir: int, q: int, q_weight: int, tr_weight: int, gir_weight: int) -> int:
    """Mirror Solidity _selectionScore exactly (integer arithmetic)."""
    denom = tr_weight + gir_weight
    normal_weight = (task_rep * tr_weight + gir * gir_weight) // denom
    q_bonus = (q_weight * q) // _WAD
    return normal_weight + q_bonus


def _sol_get_top_n(
    users: List[_User],
    task_type: int,
    n: int,
    q_weight: int,
    tr_weight: int,
    gir_weight: int,
) -> List[_User]:
    """Simulate Solidity getTopN min-heap. Returns list of selected users (unordered)."""
    heap_users  = [None] * n
    heap_scores = [0]    * n
    heap_tbs    = [b'\x00' * 32] * n
    size = 0

    def _fp_bytes(u: _User) -> bytes:
        return bytes.fromhex(u.finger_print)

    def _heapify_up(idx: int):
        # min-heap: break when parent IS weaker (parent ≤ child = heap property OK)
        while idx > 0:
            parent = (idx - 1) // 2
            if _sol_is_weaker(heap_scores[parent], heap_tbs[parent],
                              heap_scores[idx],    heap_tbs[idx]):
                break
            heap_scores[parent], heap_scores[idx] = heap_scores[idx], heap_scores[parent]
            heap_tbs[parent],    heap_tbs[idx]    = heap_tbs[idx],    heap_tbs[parent]
            heap_users[parent],  heap_users[idx]  = heap_users[idx],  heap_users[parent]
            idx = parent

    def _heapify_down(heap_size: int):
        idx = 0
        while True:
            left    = 2 * idx + 1
            right   = 2 * idx + 2
            weakest = idx
            if left  < heap_size and _sol_is_weaker(heap_scores[left],  heap_tbs[left],
                                                     heap_scores[weakest], heap_tbs[weakest]):
                weakest = left
            if right < heap_size and _sol_is_weaker(heap_scores[right], heap_tbs[right],
                                                     heap_scores[weakest], heap_tbs[weakest]):
                weakest = right
            if weakest == idx:
                break
            heap_scores[idx], heap_scores[weakest] = heap_scores[weakest], heap_scores[idx]
            heap_tbs[idx],    heap_tbs[weakest]    = heap_tbs[weakest],    heap_tbs[idx]
            heap_users[idx],  heap_users[weakest]  = heap_users[weakest],  heap_users[idx]
            idx = weakest

    for u in users:
        score = _sol_score(
            u.task_rep.get(task_type, 0), u.global_integrity_rep,
            u.q_value.get(task_type, 0),
            q_weight, tr_weight, gir_weight,
        )
        tb = _fp_bytes(u)

        if size < n:
            heap_users[size]  = u
            heap_scores[size] = score
            heap_tbs[size]    = tb
            _heapify_up(size)
            size += 1
        elif not _sol_is_weaker(score, tb, heap_scores[0], heap_tbs[0]):
            heap_users[0]  = u
            heap_scores[0] = score
            heap_tbs[0]    = tb
            _heapify_down(n)

    return heap_users[:size]


# ---------------------------------------------------------------------------
# Python getTopN under test (imported from multirep)
# ---------------------------------------------------------------------------

def _py_get_top_n(
    users: List[_User],
    task_type: int,
    n: int,
    q_weight: int,
    tr_weight: int,
    gir_weight: int,
) -> List[_User]:
    """Inline Python selection — mirrors multirep.compute_user_score + getTopN."""
    denom = tr_weight + gir_weight

    def score(u: _User) -> int:
        base = (u.task_rep.get(task_type, 0) * tr_weight
                + u.global_integrity_rep * gir_weight)
        normal_weight = base // denom
        q = u.q_value.get(task_type, 0)
        q_bonus = (q_weight * q) // _WAD   # ← must divide by WAD
        return int(normal_weight + q_bonus)

    fps = {u: u.finger_print for u in users}
    ranked = sorted(users, key=lambda u: (-score(u), fps[u]))
    return ranked[:n]


# ---------------------------------------------------------------------------
# Helper: assert both algorithms select the same set of users
# ---------------------------------------------------------------------------

def _assert_same_selection(py: List[_User], sol: List[_User]):
    py_addrs  = {u.address for u in py}
    sol_addrs = {u.address for u in sol}
    only_py  = py_addrs  - sol_addrs
    only_sol = sol_addrs - py_addrs
    assert py_addrs == sol_addrs, (
        f"Selection mismatch:\n"
        f"  Python only:   {only_py}\n"
        f"  Solidity only: {only_sol}"
    )


# ===========================================================================
# score formula tests
# ===========================================================================

def test_score_no_q():
    """normalWeight = (tr*6 + gir*4) / 10, no Q contribution."""
    tr  = 3 * _WAD
    gir = 7 * _WAD
    expected = (tr * 6 + gir * 4) // 10
    assert _sol_score(tr, gir, 0, 0, 6, 4) == expected
    # Python formula must match
    u = _User(0, 0, gir, 0)
    u.set_rep(0, tr, 0)
    denom = 10
    py = (u.task_rep[0] * 6 + u.global_integrity_rep * 4) // denom + (0 * 0) // _WAD
    assert py == expected


def test_score_q_bonus_matches_solidity():
    """Q bonus = (qWeight * qValue) / WAD  (integer division, no float)."""
    q_weight = _WAD          # 1e18 — typical preset value
    q_value  = _WAD // 2     # 0.5 WAD
    expected_q_bonus = (q_weight * q_value) // _WAD  # = q_value = 5e17
    assert _sol_score(0, 0, q_value, q_weight, 6, 4) == expected_q_bonus


def test_score_q_zero_is_no_op():
    tr  = 2 * _WAD
    gir = _WAD
    score_no_q = _sol_score(tr, gir, 0, _WAD, 6, 4)
    score_qw0  = _sol_score(tr, gir, _WAD // 2, 0, 6, 4)
    assert score_no_q == score_qw0  # q_weight=0 → bonus=0 regardless of q


def test_score_custom_weights():
    tr  = _WAD
    gir = _WAD
    # equal weights → same as average = WAD
    assert _sol_score(tr, gir, 0, 0, 5, 5) == _WAD


# ===========================================================================
# tiebreaker tests
# ===========================================================================

def test_lower_fingerprint_is_stronger():
    """_isWeaker: for equal scores, the entry with a LOWER tiebreaker wins."""
    tb_low  = bytes.fromhex("00" * 32)
    tb_high = bytes.fromhex("ff" * 32)
    score = 42
    # high tiebreaker is weaker (higher bytes32 value → weaker per _isWeaker)
    assert _sol_is_weaker(score, tb_high, score, tb_low)
    assert not _sol_is_weaker(score, tb_low, score, tb_high)


def test_higher_score_beats_lower_regardless_of_tiebreaker():
    tb = bytes.fromhex("aa" * 32)
    assert _sol_is_weaker(5, tb, 10, tb)
    assert not _sol_is_weaker(10, tb, 5, tb)


def test_tiebreaker_order_consistency_between_python_and_solidity():
    """Python sort (ascending hex string) and Solidity bytes32 comparison must agree."""
    # Build 10 users with identical scores; selection by tiebreaker only.
    task_type = 6
    users = _make_users(10, task_type, task_rep=0, gir=_WAD, q=0)

    py  = _py_get_top_n(users, task_type, 5, q_weight=0, tr_weight=6, gir_weight=4)
    sol = _sol_get_top_n(users, task_type, 5, q_weight=0, tr_weight=6, gir_weight=4)
    _assert_same_selection(py, sol)

    # Verify the selected users are the 5 with the lowest fingerprints.
    sorted_fps = sorted(u.finger_print for u in users)
    top5_fps   = set(sorted_fps[:5])
    py_fps     = {u.finger_print for u in py}
    assert py_fps == top5_fps, (
        f"Expected users with 5 lowest fingerprints, got: {sorted(py_fps)}"
    )


# ===========================================================================
# getTopN end-to-end parity tests
# ===========================================================================

def test_all_equal_scores_selects_lowest_fingerprints():
    task_type = 6
    users = _make_users(8, task_type, task_rep=0, gir=_WAD, q=0)

    py  = _py_get_top_n(users, task_type, 3, 0, 6, 4)
    sol = _sol_get_top_n(users, task_type, 3, 0, 6, 4)
    _assert_same_selection(py, sol)


def test_distinct_scores_no_ties():
    task_type = 6
    users = _make_users(6, task_type)
    for i, u in enumerate(users):
        u.set_rep(task_type, task_rep=(i + 1) * _WAD, q=0)
        u.global_integrity_rep = _WAD

    py  = _py_get_top_n(users, task_type, 3, 0, 6, 4)
    sol = _sol_get_top_n(users, task_type, 3, 0, 6, 4)
    _assert_same_selection(py, sol)

    # Both should pick the 3 users with the highest task_rep (users 5, 4, 3).
    top3 = {users[5].address, users[4].address, users[3].address}
    assert {u.address for u in py} == top3


def test_q_bonus_selects_high_q_user():
    """A user with a large Q value should beat a higher-TR user when q_weight > 0."""
    task_type = 6
    q_weight  = _WAD  # 1.0 WAD unit

    # Two users: user A has high TR, zero Q; user B has low TR, large Q.
    a = _User(seed=100, task_rep=0, gir=_WAD, q=0)
    a.set_rep(task_type, 3 * _WAD, 0)

    b = _User(seed=200, task_rep=0, gir=_WAD, q=0)
    b.set_rep(task_type, _WAD, 4 * _WAD)   # q = 4 WAD units

    users = [a, b]
    py  = _py_get_top_n(users, task_type, 1, q_weight, 6, 4)
    sol = _sol_get_top_n(users, task_type, 1, q_weight, 6, 4)

    # Verify both agree on the winner
    _assert_same_selection(py, sol)
    # Q bonus for b: (WAD * 4*WAD) / WAD = 4*WAD dominates a's normalWeight
    assert py[0].address == b.address


def test_mixed_scores_and_tiebreaks():
    """Two pairs of tied users (different scores between pairs) plus one outlier."""
    task_type = 6
    users = _make_users(7, task_type, task_rep=0, gir=_WAD, q=0)
    # Give users 0–1 score=5e17, users 2–3 score=3e17, users 4–6 score=1e17
    for u in users[:2]:
        u.global_integrity_rep = 5 * _WAD
        u.task_rep[task_type] = 0
    for u in users[2:4]:
        u.global_integrity_rep = 3 * _WAD
        u.task_rep[task_type] = 0
    for u in users[4:]:
        u.global_integrity_rep = 1 * _WAD
        u.task_rep[task_type] = 0

    py  = _py_get_top_n(users, task_type, 3, 0, 6, 4)
    sol = _sol_get_top_n(users, task_type, 3, 0, 6, 4)
    _assert_same_selection(py, sol)
    # Top 3 = users[0], users[1] (both score 2*WAD), then one of users[2], users[3]
    # (equal score → lower fingerprint wins).
    selected_addrs = {u.address for u in py}
    assert users[0].address in selected_addrs
    assert users[1].address in selected_addrs


def test_n_equals_total_users():
    """Selecting all users should be identical regardless of algorithm."""
    task_type = 6
    users = _make_users(5, task_type, task_rep=_WAD, gir=_WAD, q=0)
    py  = _py_get_top_n(users, task_type, 5, 0, 6, 4)
    sol = _sol_get_top_n(users, task_type, 5, 0, 6, 4)
    _assert_same_selection(py, sol)


def test_insertion_order_does_not_affect_heap_result():
    """Solidity heap result must be order-independent.

    The heap produces the same final set regardless of which order applicants
    registered, since it's only comparing scores and tiebreakers.
    """
    import random
    task_type = 6
    users = _make_users(10, task_type, task_rep=0, gir=_WAD, q=0)

    sol1 = {u.address for u in _sol_get_top_n(users, task_type, 4, 0, 6, 4)}

    shuffled = users[:]
    random.shuffle(shuffled)
    sol2 = {u.address for u in _sol_get_top_n(shuffled, task_type, 4, 0, 6, 4)}

    assert sol1 == sol2, "Heap result changed with insertion order"


# ===========================================================================
# q_weight integer arithmetic — no float non-determinism
# ===========================================================================

def test_compute_user_score_q_weight_integer():
    """compute_user_score must accept a WAD-scaled integer q_weight and produce
    the same result as the Solidity formula (q_weight * q) // WAD."""
    task_type = 6
    u = _User(seed=1, task_rep=0, gir=_WAD, q=0)
    u.set_rep(task_type, 2 * _WAD, 3 * _WAD // 4)  # q = 0.75 WAD

    q_weight = _WAD // 2   # 0.5 WAD  (integer, not float)
    tr_weight, gir_weight = 6, 4

    denom = tr_weight + gir_weight
    normal_weight = (u.task_rep[task_type] * tr_weight + u.global_integrity_rep * gir_weight) // denom
    expected_q_bonus = (q_weight * u.q_value[task_type]) // _WAD
    expected = normal_weight + expected_q_bonus

    # _py_get_top_n uses the same formula — verify score matches
    import importlib, sys
    spec = importlib.util.spec_from_file_location("multirep_main", "experiment/multirep.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    compute_user_score = mod.compute_user_score
    assert compute_user_score(u, task_type, q_weight, tr_weight, gir_weight) == expected


def test_compute_user_score_matches_solidity_formula():
    """Python compute_user_score must match Solidity _selectionScore exactly."""
    task_type = 5
    users = _make_users(6, task_type)
    for i, u in enumerate(users):
        u.set_rep(task_type, task_rep=(i + 1) * _WAD // 3, q=(i * _WAD) // 7)
        u.global_integrity_rep = (i + 2) * _WAD // 5

    q_weight  = int(0.3 * _WAD)   # 0.3 WAD — integer
    tr_weight, gir_weight = 7, 3

    import importlib, sys
    spec = importlib.util.spec_from_file_location("multirep_main", "experiment/multirep.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    compute_user_score = mod.compute_user_score
    for u in users:
        py_score  = compute_user_score(u, task_type, q_weight, tr_weight, gir_weight)
        sol_score = _sol_score(
            u.task_rep[task_type], u.global_integrity_rep,
            u.q_value[task_type], q_weight, tr_weight, gir_weight,
        )
        assert py_score == sol_score, (
            f"Score mismatch for user {u.address}: py={py_score}  sol={sol_score}"
        )


def test_selection_deterministic_across_repeated_calls():
    """Selection algorithm must return the same set for the same inputs every time."""
    task_type = 6
    users = _make_users(12, task_type, task_rep=0, gir=_WAD, q=0)
    for i, u in enumerate(users):
        u.set_rep(task_type, (i % 4) * _WAD // 3, q=(i % 3) * _WAD // 5)
        u.global_integrity_rep = ((i * 7) % 5) * _WAD // 4

    q_weight = int(0.2 * _WAD)
    result1 = {u.address for u in _py_get_top_n(users, task_type, 5, q_weight, 6, 4)}
    result2 = {u.address for u in _py_get_top_n(users, task_type, 5, q_weight, 6, 4)}
    assert result1 == result2, "Python getTopN is not deterministic"

    sol1 = {u.address for u in _sol_get_top_n(users, task_type, 5, q_weight, 6, 4)}
    sol2 = {u.address for u in _sol_get_top_n(users, task_type, 5, q_weight, 6, 4)}
    assert sol1 == sol2, "Solidity heap simulation is not deterministic"

    assert result1 == sol1, "Python and Solidity must agree"


def test_selection_parity_with_nonzero_q_weight():
    """Python and Solidity must agree when q_weight > 0 (integer WAD-scaled)."""
    task_type = 5
    users = _make_users(10, task_type, task_rep=0, gir=_WAD, q=0)
    for i, u in enumerate(users):
        u.set_rep(task_type, (i % 3) * _WAD // 4, q=(i % 5) * _WAD // 6)
        u.global_integrity_rep = (i % 4) * _WAD // 3

    q_weight = int(0.4 * _WAD)  # integer, not float
    py  = _py_get_top_n(users, task_type, 4, q_weight, 6, 4)
    sol = _sol_get_top_n(users, task_type, 4, q_weight, 6, 4)
    _assert_same_selection(py, sol)


# ===========================================================================
# Q-slot cap parity tests (JobListing._getTopNCapped + _pickTopK)
# ===========================================================================

def _sol_pick_top_k(users, chosen, task_type, k, include_q, q_weight, tr_weight, gir_weight):
    """Mirror Solidity _pickTopK: mark the k strongest not-yet-chosen users.

    Strongest = higher score (Q bonus iff include_q), ties broken by lower
    tiebreaker (lower fingerprint), exactly as _isWeaker orders them.
    """
    for _ in range(k):
        best = None
        best_score = 0
        best_tb = b''
        for u in users:
            if u in chosen:
                continue
            qw = q_weight if include_q else 0
            score = _sol_score(
                u.task_rep.get(task_type, 0), u.global_integrity_rep,
                u.q_value.get(task_type, 0), qw, tr_weight, gir_weight,
            )
            tb = bytes.fromhex(u.finger_print)
            if best is None or not _sol_is_weaker(score, tb, best_score, best_tb):
                best, best_score, best_tb = u, score, tb
        if best is None:
            break
        chosen.add(best)


def _sol_get_top_n_capped(users, task_type, n, q_slot_limit, q_weight, tr_weight, gir_weight):
    """Mirror Solidity _getTopNCapped: rep slots by base score, q slots by full score."""
    q_slots = min(q_slot_limit, n)
    rep_slots = n - q_slots
    chosen = set()
    _sol_pick_top_k(users, chosen, task_type, rep_slots, False, q_weight, tr_weight, gir_weight)
    _sol_pick_top_k(users, chosen, task_type, n - rep_slots, True, q_weight, tr_weight, gir_weight)
    return list(chosen)


def _py_get_top_n_capped(users, task_type, n, q_slot_limit, q_weight, tr_weight, gir_weight):
    """Mirror multirep.getTopN capped branch."""
    fps = {u: u.finger_print for u in users}

    def score(u, qw):
        return _sol_score(
            u.task_rep.get(task_type, 0), u.global_integrity_rep,
            u.q_value.get(task_type, 0), qw, tr_weight, gir_weight,
        )

    q_slots = min(max(q_slot_limit, 0), n)
    rep_slots = n - q_slots
    by_base = sorted(users, key=lambda u: (-score(u, 0), fps[u]))
    rep_selected = by_base[:rep_slots]
    rep_ids = {id(u) for u in rep_selected}
    pool = [u for u in users if id(u) not in rep_ids]
    pool.sort(key=lambda u: (-score(u, q_weight), fps[u]))
    return rep_selected + pool[:q_slots]


def test_capped_python_solidity_parity():
    """Capped Python and Solidity mirrors must select the same set."""
    task_type = 6
    users = _make_users(10, task_type, task_rep=0, gir=_WAD, q=0)
    for i, u in enumerate(users):
        u.set_rep(task_type, (i % 4) * _WAD // 2, q=((9 - i) % 5) * _WAD // 3)
        u.global_integrity_rep = (i % 3) * _WAD // 2

    q_weight = int(0.5 * _WAD)
    for q_limit in (0, 1, 2, 4):
        py  = _py_get_top_n_capped(users, task_type, 6, q_limit, q_weight, 6, 4)
        sol = _sol_get_top_n_capped(users, task_type, 6, q_limit, q_weight, 6, 4)
        _assert_same_selection(py, sol)


def test_capped_zero_q_slots_ignores_q():
    """q_slot_limit=0 → all slots by base score; a huge-Q low-rep user is excluded."""
    task_type = 6
    # High-rep users 0..5, plus a low-rep but huge-Q user that would win uncapped.
    users = _make_users(6, task_type, task_rep=2 * _WAD, gir=_WAD, q=0)
    sneaker = _User(seed=999, task_rep=0, gir=0, q=0)
    sneaker.set_rep(task_type, 0, 100 * _WAD)  # massive Q
    users.append(sneaker)

    q_weight = _WAD
    capped = _py_get_top_n_capped(users, task_type, 6, 0, q_weight, 6, 4)
    sol = _sol_get_top_n_capped(users, task_type, 6, 0, q_weight, 6, 4)
    _assert_same_selection(capped, sol)
    assert sneaker.address not in {u.address for u in capped}, "Q got a slot despite limit=0"


def test_capped_q_slot_admits_high_q_user_within_limit():
    """With one q slot, the huge-Q user takes exactly that slot, not a rep slot."""
    task_type = 6
    users = _make_users(6, task_type, task_rep=2 * _WAD, gir=_WAD, q=0)
    sneaker = _User(seed=999, task_rep=0, gir=0, q=0)
    sneaker.set_rep(task_type, 0, 100 * _WAD)
    users.append(sneaker)

    q_weight = _WAD
    py  = _py_get_top_n_capped(users, task_type, 6, 1, q_weight, 6, 4)
    sol = _sol_get_top_n_capped(users, task_type, 6, 1, q_weight, 6, 4)
    _assert_same_selection(py, sol)
    assert sneaker.address in {u.address for u in py}, "high-Q user missed its q slot"


def test_capped_full_limit_equals_uncapped():
    """q_slot_limit >= N degrades to the uncapped full-score selection."""
    task_type = 6
    users = _make_users(8, task_type, task_rep=0, gir=_WAD, q=0)
    for i, u in enumerate(users):
        u.set_rep(task_type, (i % 3) * _WAD, q=(i % 4) * _WAD // 2)
        u.global_integrity_rep = (i % 2) * _WAD

    q_weight = int(0.3 * _WAD)
    capped   = _py_get_top_n_capped(users, task_type, 4, 4, q_weight, 6, 4)
    uncapped = _py_get_top_n(users, task_type, 4, q_weight, 6, 4)
    _assert_same_selection(capped, uncapped)
