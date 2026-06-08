"""Unit tests for the aggregate multi-experiment graph plumbing.

Covers the pure-data logic (no plotting): data-split categorisation, run
averaging, experiment pairing, curve aggregation, session-third splitting, and
partition-file loading.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from analysis.multirep_aggregate_loader import (  # noqa: E402
    ExperimentRuns,
    MultirepSession,
    build_pairs,
    load_partition_data_percent,
    _pair_key,
)
from analysis import multirep_aggregate_plots as ap  # noqa: E402
from analysis import multirep_grouped_plots as gp  # noqa: E402
from analysis import multirep_runavg as ra  # noqa: E402
from analysis import multirep_plots as mrp  # noqa: E402
from analysis import multirep_thesis_plots as tp  # noqa: E402


# ---------------------------------------------------------------------------
# split_category — all three preset naming schemes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("H1 MNIST-heavy, expert MNIST 0-1 / CIFAR 4-5", "MNIST-heavy"),
    ("H5 CIFAR-heavy, expert MNIST 8-9 / CIFAR 2-3", "CIFAR-heavy"),
    ("H8 Balanced, expert MNIST 4-5 / CIFAR 8-9", "Balanced"),
    ("MNIST-strong 3", "MNIST-strong"),
    ("CIFAR-strong 5", "CIFAR-strong"),
    ("Average 8", "Average"),
    ("H3 Both", "Both"),
    ("F2 MNIST", "MNIST-only"),
    ("M3 CIFAR-10", "CIFAR-only"),
    ("Weirdo 9", "Other"),
])
def test_split_category(name, expected):
    assert gp.split_category(name) == expected


def test_split_dataset_bias():
    assert gp.split_dataset_bias("MNIST-strong") == 5
    assert gp.split_dataset_bias("CIFAR-heavy") == 6
    assert gp.split_dataset_bias("Balanced") is None
    assert gp.split_dataset_bias("Average") is None


# ---------------------------------------------------------------------------
# _pair_key + build_pairs
# ---------------------------------------------------------------------------

def test_pair_key_strips_system_token():
    assert _pair_key("EXP-multirep-avg-distribution-5-task", "multirep") == "exp-avg-distribution-5-task"
    assert _pair_key("EXP-globalrep-avg-distribution-5-task", "globalrep") == "exp-avg-distribution-5-task"
    # a variant token stays in the key so it does NOT pair with the plain one
    assert _pair_key("EXP-multirep-noqvalue-avg", "multirep") != _pair_key("EXP-multirep-avg", "multirep")


def _exp(name, system, key):
    return ExperimentRuns(name=name, system=system, pair_key=key, sessions=[])


def test_build_pairs_groups_systems():
    exps = [
        _exp("EXP-globalrep-avg", "globalrep", "exp-avg"),
        _exp("EXP-multirep-avg", "multirep", "exp-avg"),
        _exp("EXP-multirep-noqvalue-avg", "multirep", "exp-noqvalue-avg"),
    ]
    pairs = {p.key: p for p in build_pairs(exps)}
    assert pairs["exp-avg"].is_complete()
    assert pairs["exp-avg"].globalrep.name == "EXP-globalrep-avg"
    assert pairs["exp-avg"].multirep.name == "EXP-multirep-avg"
    # noqvalue is multirep-only → not a complete pair
    assert not pairs["exp-noqvalue-avg"].is_complete()


# ---------------------------------------------------------------------------
# _curve and _thirds
# ---------------------------------------------------------------------------

def test_curve_mean_band():
    df = pd.DataFrame({"round": [0, 0, 1, 1], "v": [0.0, 1.0, 2.0, 4.0]})
    x, c, lo, hi = ap._curve(df, "v", "mean")
    assert list(x) == [0, 1]
    assert c[0] == pytest.approx(0.5) and c[1] == pytest.approx(3.0)
    # band is ±1 std (ddof=1): std of [0,1]=0.707
    assert lo[0] == pytest.approx(0.5 - 0.7071, abs=1e-3)


def test_curve_median_iqr():
    df = pd.DataFrame({"round": [0] * 4, "v": [1.0, 2.0, 3.0, 4.0]})
    x, c, lo, hi = ap._curve(df, "v", "median")
    assert c[0] == pytest.approx(2.5)
    assert lo[0] == pytest.approx(1.75) and hi[0] == pytest.approx(3.25)


def test_thirds_splits_contiguously():
    assert ap._thirds([0, 1, 2, 3, 4, 5]) == [[0, 1], [2, 3], [4, 5]]
    # uneven split: numpy puts the remainder in the earlier buckets
    assert ap._thirds([1, 2, 3, 4]) == [[1, 2], [3], [4]]
    assert ap._thirds([]) == [[], [], []]


# ---------------------------------------------------------------------------
# run averaging
# ---------------------------------------------------------------------------

def _rep_two_runs():
    rows = []
    for run, (tr, sel) in enumerate([(0.2, True), (0.4, False)]):
        rows.append({
            "guid": "g1", "task_index": 0, "user_name": "MNIST-strong 1",
            "behavior": "honest", "run": run, "tr_post": tr, "gir_post": 0.5,
            "was_selected": sel, "tr_all_post": {5: tr, 6: 0.0},
        })
    return pd.DataFrame(rows)


def test_average_runs_scalar_and_dicts():
    avg = ra.average_runs(_rep_two_runs())
    assert len(avg) == 1
    row = avg.iloc[0]
    assert row["tr_post"] == pytest.approx(0.3)         # (0.2+0.4)/2
    assert row["tr_all_post"][5] == pytest.approx(0.3)  # dict averaged per key
    assert row["selection_freq"] == pytest.approx(0.5)  # selected in 1 of 2 runs


def test_average_runs_was_selected_threshold():
    df = _rep_two_runs()
    # freq 0.5 → not strictly >= 0.5? it is exactly 0.5 → True
    assert bool(ra.average_runs(df).iloc[0]["was_selected"]) is True
    # make it 0/2 → False
    df2 = df.copy(); df2["was_selected"] = [False, False]
    assert bool(ra.average_runs(df2).iloc[0]["was_selected"]) is False


def test_average_global_accuracy_means_over_runs():
    ga = pd.DataFrame({
        "run": [0, 1, 0, 1],
        "task_index": [0, 0, 0, 0],
        "dataset": ["mnist"] * 4,
        "round": [0, 0, 1, 1],
        "objective_global_accuracy": [0.1, 0.3, 0.8, 0.9],
        "objective_global_loss": [2.0, 2.0, 0.5, 0.5],
    })
    out = ra.average_global_accuracy(ga)
    r0 = out[out["round"] == 0]["objective_global_accuracy"].iloc[0]
    r1 = out[out["round"] == 1]["objective_global_accuracy"].iloc[0]
    assert r0 == pytest.approx(0.2) and r1 == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# partition loading (uses a real partition file from the repo)
# ---------------------------------------------------------------------------

def _session_with_partition(pf: str) -> MultirepSession:
    return MultirepSession(
        session_id="s", preset_name="p", session_timestamp="t",
        preset={"partition_file": pf},
        reputation_timeline=pd.DataFrame(), global_accuracy=pd.DataFrame(), tasks=[],
    )


def test_load_partition_data_percent_mixed():
    pf = "experiment/partitions/EXP-mixed-distribution-all-honest-20-users.json"
    if not (_REPO / pf).exists():
        pytest.skip("partition file not present")
    exp = ExperimentRuns(name="EXP-multirep-mixed", system="multirep",
                         pair_key="k", sessions=[_session_with_partition(pf)])
    pct = load_partition_data_percent(exp)
    assert pct, "should parse at least one participant"
    # MNIST-strong users hold more MNIST (tt=5) than CIFAR (tt=6) data
    strong = next(v for k, v in pct.items() if "mnist-strong" in k.lower())
    assert strong[5] > strong[6]


def test_load_partition_missing_file_returns_empty():
    exp = ExperimentRuns(name="x", system="multirep", pair_key="k",
                         sessions=[_session_with_partition("nope/missing.json")])
    assert load_partition_data_percent(exp) == {}


# ---------------------------------------------------------------------------
# per-dataset earnings attribution (mixed-behavior users)
# ---------------------------------------------------------------------------

def _exp_from_rep(rep: pd.DataFrame) -> ExperimentRuns:
    sess = MultirepSession(
        session_id="s", preset_name="p", session_timestamp="t", preset={},
        reputation_timeline=rep, global_accuracy=pd.DataFrame(), tasks=[],
    )
    return ExperimentRuns(name="EXP-multirep-x", system="multirep",
                          pair_key="k", sessions=[sess])


def _mixed_rep() -> pd.DataFrame:
    # U1: free-rider on MNIST(tt5, +2), honest on CIFAR(tt6, +1)
    # U2: honest on both
    return pd.DataFrame([
        dict(user_name="MNIST-strong 1", task_index=0, task_type=5,
             behavior="freerider", balance_pre=0.0, balance_post=2.0),
        dict(user_name="MNIST-strong 1", task_index=1, task_type=6,
             behavior="honest", balance_pre=2.0, balance_post=3.0),
        dict(user_name="Average 2", task_index=0, task_type=5,
             behavior="honest", balance_pre=0.0, balance_post=1.0),
        dict(user_name="Average 2", task_index=1, task_type=6,
             behavior="honest", balance_pre=1.0, balance_post=2.5),
    ])


def test_per_task_delta_attribution_no_double_count():
    rep = gp._per_task_delta(gp._with_category(_mixed_rep()))
    # group exactly like plot_net_earnings_by_split
    per = rep.groupby(["user_name", "behavior"])["_delta"].sum()
    # U1's free-rider part (+2) and honest part (+1) sum to its true total (+3)
    assert per[("MNIST-strong 1", "freerider")] == pytest.approx(2.0)
    assert per[("MNIST-strong 1", "honest")] == pytest.approx(1.0)
    total_u1 = rep[rep["user_name"] == "MNIST-strong 1"]["_delta"].sum()
    assert total_u1 == pytest.approx(3.0)  # no double count


def test_mixed_user_table_picks_only_mixed():
    exp = _exp_from_rep(_mixed_rep())
    t = gp._mixed_user_table(exp)
    assert set(t["user_name"]) == {"MNIST-strong 1"}  # U2 is honest on both → excluded
    u1 = t.set_index("task_type")
    assert u1.loc[5, "behavior"] == "freerider" and u1.loc[5, "delta"] == pytest.approx(2.0)
    assert u1.loc[6, "behavior"] == "honest" and u1.loc[6, "delta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# variance honesty: fingerprint cache-clone dedupe
# ---------------------------------------------------------------------------

def test_distinct_curves_collapses_fingerprint_clones():
    # two task slots share fingerprint "a" (a cache clone) + one distinct "b"
    df = pd.DataFrame({
        "run": [0, 0, 0, 0],
        "fingerprint": ["a", "a", "b", "b"],
        "round": [0, 0, 0, 0],
        "task_index": [0, 1, 2, 2],
        "v": [0.5, 0.5, 0.9, 0.9],
    })
    out = ap._distinct_curves(df)
    assert len(out) == 2                      # one row per (run, fingerprint, round)
    assert ap._n_curves(df) == 2              # 2 distinct (run, fingerprint) curves


def test_curve_ignores_duplicated_fingerprint_for_std():
    # without dedupe the clone would shrink std toward 0; with dedupe std reflects
    # the two genuinely-distinct curves only.
    df = pd.DataFrame({
        "run": [0, 0, 0],
        "fingerprint": ["a", "a", "b"],   # 'a' duplicated
        "round": [0, 0, 0],
        "v": [0.2, 0.2, 0.8],
    })
    _x, c, _lo, _hi = ap._curve(df, "v", "mean")
    assert c[0] == pytest.approx(0.5)         # mean of the two distinct curves (0.2, 0.8)


# ---------------------------------------------------------------------------
# selected-only progression (x = n-th selection)
# ---------------------------------------------------------------------------

def test_selected_reindex_drops_unselected_and_renumbers():
    rep = pd.DataFrame([
        dict(run=0, user_name="U", behavior="honest", task_index=0, was_selected=True, tr_post=0.1),
        dict(run=0, user_name="U", behavior="honest", task_index=1, was_selected=False, tr_post=0.1),
        dict(run=0, user_name="U", behavior="honest", task_index=2, was_selected=True, tr_post=0.3),
    ])
    out = mrp._selected_reindex(rep, "tr_post")
    assert list(out["nth"]) == [1, 2]                 # only the two selected tasks
    assert out.loc[out["nth"] == 2, "tr_post"].iloc[0] == pytest.approx(0.3)


def test_selected_reindex_averages_across_runs():
    rep = pd.DataFrame([
        dict(run=0, user_name="U", behavior="honest", task_index=0, was_selected=True, tr_post=0.2),
        dict(run=1, user_name="U", behavior="honest", task_index=0, was_selected=True, tr_post=0.4),
    ])
    out = mrp._selected_reindex(rep, "tr_post")
    assert out.loc[out["nth"] == 1, "tr_post"].iloc[0] == pytest.approx(0.3)  # mean(0.2,0.4)


# ---------------------------------------------------------------------------
# thesis helpers: spearman, welch, cross-task TR
# ---------------------------------------------------------------------------

def test_spearman_monotonic_and_degenerate():
    assert tp._spearman(np.array([1, 2, 3, 4]), np.array([2, 4, 6, 8])) == pytest.approx(1.0)
    assert tp._spearman(np.array([1, 2, 3, 4]), np.array([8, 6, 4, 2])) == pytest.approx(-1.0)
    assert np.isnan(tp._spearman(np.array([1, 1, 1]), np.array([1, 2, 3])))  # no spread


def test_welch_p_separated_vs_identical():
    near0 = tp._welch_p(np.array([10.0, 10.1, 9.9]), np.array([0.0, 0.1, -0.1]))
    assert near0 < 0.05                       # clearly separated means
    high = tp._welch_p(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert high == pytest.approx(1.0, abs=1e-6)  # identical → no difference


def test_final_tr_per_user_globalrep_on_diagonal():
    rep = pd.DataFrame([
        dict(run=0, user_name="U", behavior="honest", task_index=0, tr_post=0.6,
             tr_all_post={5: 0.6, 6: 0.6}),
    ])
    exp = ExperimentRuns(name="EXP-globalrep-x", system="globalrep", pair_key="k",
                         sessions=[MultirepSession(session_id="s", preset_name="p",
                         session_timestamp="t", preset={}, reputation_timeline=rep,
                         global_accuracy=pd.DataFrame(), tasks=[])])
    df = tp._final_tr_per_user(exp)
    assert df["tr_mnist"].iloc[0] == df["tr_cifar"].iloc[0] == pytest.approx(0.6)


def test_final_tr_per_user_multirep_splits_task_types():
    rep = pd.DataFrame([
        dict(run=0, user_name="U", behavior="honest", task_index=0, tr_post=0.0,
             tr_all_post={5: 0.5, 6: 0.1}),
    ])
    df = tp._final_tr_per_user(_exp_from_rep(rep))
    assert df["tr_mnist"].iloc[0] == pytest.approx(0.5)
    assert df["tr_cifar"].iloc[0] == pytest.approx(0.1)
