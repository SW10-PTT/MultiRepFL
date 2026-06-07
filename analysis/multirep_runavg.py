"""Average multiple runs of one experiment into a single session-shaped view.

Lets the existing single-run plot functions in ``multirep_plots`` be reused for
multi-run data: collapse the runs of an :class:`ExperimentRuns` into one
reputation_timeline and one global_accuracy table by averaging matching rows.

Matching is by (guid, task_index) for reputation rows and by
(task_index, dataset, round) for accuracy rows — both stable across runs since
all runs of an experiment share the same preset.
"""

from __future__ import annotations

import pandas as pd

from analysis.multirep_aggregate_loader import ExperimentRuns

# Scalar numeric columns averaged across runs.
_NUM_COLS = [
    "tr_pre", "tr_post", "gir_pre", "gir_post", "q_pre", "q_post",
    "balance_pre", "balance_post", "selection_score", "total_contrib_post",
    "contrib_score", "confidence", "k", "running_c_mean", "m2",
]
# Dict columns averaged element-wise (per task-type key).
_DICT_COLS = ["tr_all_pre", "tr_all_post", "q_all_pre", "q_all_post"]


def _avg_dicts(series) -> dict:
    sums: dict = {}
    cnt: dict = {}
    for d in series:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            sums[k] = sums.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1
    return {k: sums[k] / cnt[k] for k in sums}


def average_runs(rep: pd.DataFrame) -> pd.DataFrame:
    """Collapse a run-tagged reputation timeline to one row per (guid, task_index).

    Numeric columns are mean-averaged across runs; ``was_selected`` becomes the
    selection *frequency* in [0, 1]; dict columns are averaged per key.
    """
    if rep.empty:
        return rep
    rows = []
    for (_guid, _ti), g in rep.groupby(["guid", "task_index"], sort=False):
        row = g.iloc[0].to_dict()
        for c in _NUM_COLS:
            if c in g.columns:
                row[c] = pd.to_numeric(g[c], errors="coerce").mean()
        for c in _DICT_COLS:
            if c in g.columns:
                row[c] = _avg_dicts(g[c])
        if "was_selected" in g.columns:
            freq = g["was_selected"].astype(float).mean()
            row["selection_freq"] = freq          # selection frequency across runs
            row["was_selected"] = bool(freq >= 0.5)  # bool for the single-run plot fns
        rows.append(row)
    return pd.DataFrame(rows)


def average_global_accuracy(ga: pd.DataFrame) -> pd.DataFrame:
    """Average per-round accuracy/loss across runs, keyed by (task_index, dataset, round)."""
    if ga.empty:
        return ga
    value_cols = [c for c in (
        "round_time", "objective_global_accuracy", "objective_global_loss",
        "reward_pool", "punishment_pool",
    ) if c in ga.columns]
    return (
        ga.groupby(["task_index", "dataset", "round"], as_index=False)[value_cols]
        .mean(numeric_only=True)
    )


def averaged_views(exp: ExperimentRuns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reputation_timeline_avg, global_accuracy_avg) for one experiment."""
    return average_runs(exp.reputation_timeline()), average_global_accuracy(exp.global_accuracy())
