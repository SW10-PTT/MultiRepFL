"""Aggregate comparison plots across multiple multirep experiments / runs.

Every function takes an :class:`ExperimentPair` (globalrep vs multirep) or a
pair of :class:`ExperimentRuns` and returns a matplotlib Figure.  Runs of the
same experiment are pooled and averaged so the curves are stable.

Conventions
-----------
* System  → line *colour* + *style* (multirep solid, globalrep dashed) in
  system-level graphs.
* Behavior → colour (honest/malicious/freerider) in participant-level graphs;
  system is then distinguished by line style or bar hatch.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from analysis.multirep_aggregate_loader import (
    CIFAR_TT,
    MNIST_TT,
    TASK_TYPE_LABELS,
    ExperimentPair,
    ExperimentRuns,
)
from analysis.multirep_plots import BEHAVIOR_COLORS, BEHAVIOR_LABELS

matplotlib.rcParams.update({"figure.dpi": 200})

# --- styling -------------------------------------------------------------
SYSTEM_COLORS = {"multirep": "#1b9e77", "globalrep": "#7570b3"}
SYSTEM_LS = {"multirep": "-", "globalrep": "--"}
SYSTEM_LABELS = {"multirep": "Multi-rep", "globalrep": "Global-rep"}
SYSTEM_HATCH = {"multirep": "", "globalrep": "//"}

BEHAVIOR_ORDER = ["honest", "malicious", "freerider"]

# task_type → dataset string used in the global_accuracy table
TT_DATASET = {MNIST_TT: "mnist", CIFAR_TT: "cifar-10"}

# time-to-accuracy targets
ACC_THRESHOLDS = {"mnist": 0.95, "cifar-10": 0.45, "cifar10": 0.45}
ACC_FRACTION = 0.9

# Max FL round shown on round-based curves, per dataset (later rounds are flat).
ROUND_CAP = {MNIST_TT: 10, CIFAR_TT: 15}

# Light background tints marking which dataset each task ran (over-task graphs).
DATASET_TINT = {MNIST_TT: "#2196F3", CIFAR_TT: "#FF9800"}

_LW = 2
_FILL_ALPHA = 0.15
_EPS = 1e-6


# =========================================================================
# small aggregation helpers
# =========================================================================

def _distinct_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fingerprint cache-hit clones so variance bands reflect real spread.

    The loader back-fills every task slot from the one computed curve sharing its
    fingerprint, so identical (run, fingerprint, round) rows repeat.  Collapsing
    them to one row per (run, fingerprint, round) keeps the mean unchanged but
    stops the duplicated curves from artificially shrinking std/IQR.
    """
    if {"run", "fingerprint", "round"} <= set(df.columns):
        return df.drop_duplicates(["run", "fingerprint", "round"])
    return df


def _behavior_on_task_type(rep: pd.DataFrame, task_type: int) -> pd.Series:
    """Each row's user behaviour *on the given task type* (constant per user),
    aligned back to ``rep``'s index.  Task-hoppers carry a different behaviour per
    dataset, so the per-task ``behavior`` column flips at dataset switches; this
    fixes a user to its behaviour on one dataset for coherent per-dataset grouping.
    """
    sub = rep[rep["task_type"] == task_type]
    if sub.empty:
        # no tasks of this type logged → fall back to the per-task label
        return rep["behavior"]
    by_user = sub.groupby("user_name")["behavior"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
    return rep["user_name"].map(by_user)


def _n_curves(df: pd.DataFrame) -> int:
    """Number of distinct (run, fingerprint) experimental curves in *df*."""
    if {"run", "fingerprint"} <= set(df.columns):
        return int(df.drop_duplicates(["run", "fingerprint"]).shape[0])
    if {"run", "task_index"} <= set(df.columns):
        return int(df.drop_duplicates(["run", "task_index"]).shape[0])
    return len(df)


def _curve(df: pd.DataFrame, col: str, method: str):
    """Aggregate *col* over 'round'. Returns (x, center, lo, hi).

    method='mean'   → center=mean, band=±1 std
    method='median' → center=median, band=[Q1, Q3]
    """
    df = _distinct_curves(df)
    g = df.groupby("round")[col]
    if method == "median":
        center = g.median()
        lo = g.quantile(0.25)
        hi = g.quantile(0.75)
    else:
        center = g.mean()
        sd = g.std().fillna(0.0)
        lo = center - sd
        hi = center + sd
    x = center.index.values
    return x, center.values, lo.values, hi.values


def _dataset_indices(pair: ExperimentPair, dataset: str) -> list[int]:
    """Sorted unique task indices for *dataset*, unioned across both systems."""
    idx: set[int] = set()
    for _, exp in pair.items():
        ga = exp.global_accuracy()
        if ga.empty:
            continue
        sub = ga[ga["dataset"].str.lower() == dataset]
        idx |= set(int(i) for i in sub["task_index"].unique())
    return sorted(idx)


def _thirds(indices: list[int]) -> list[list[int]]:
    """Split an ordered index list into three contiguous chronological buckets."""
    if not indices:
        return [[], [], []]
    return [list(part) for part in np.array_split(np.array(indices), 3)]


def _task_dataset_map(pair: ExperimentPair) -> dict[int, int]:
    """task_index -> task_type, unioned across both systems (shared preset)."""
    mapping: dict[int, int] = {}
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if rep.empty:
            continue
        for ti, tt in rep[["task_index", "task_type"]].drop_duplicates().itertuples(index=False):
            mapping[int(ti)] = int(tt)
    return dict(sorted(mapping.items()))


def _dataset_segments(pair: ExperimentPair):
    """Contiguous (start, end, task_type) runs over task_index — for switch shading."""
    mp = _task_dataset_map(pair)
    segments = []
    items = list(mp.items())
    if not items:
        return segments
    start, cur_tt = items[0][0], items[0][1]
    prev = start
    for ti, tt in items[1:]:
        if tt != cur_tt:
            segments.append((start, prev, cur_tt))
            start, cur_tt = ti, tt
        prev = ti
    segments.append((start, prev, cur_tt))
    return segments


def _mark_dataset_switches(ax, pair: ExperimentPair, *, shade: bool = True) -> None:
    """Tint the background per dataset run and draw dashed lines at switches.

    Makes flat stretches meaningful: e.g. multirep's CIFAR task-rep stays flat
    while a run of MNIST tasks executes.
    """
    segments = _dataset_segments(pair)
    for i, (start, end, tt) in enumerate(segments):
        if shade:
            ax.axvspan(start - 0.5, end + 0.5, color=DATASET_TINT.get(tt, "#999"),
                       alpha=0.06, zorder=0)
        if i > 0:
            ax.axvline(start - 0.5, color="#555", ls=":", lw=1, alpha=0.5, zorder=1)
    # dataset legend handles for the caller to merge
    handles = [Patch(facecolor=DATASET_TINT.get(tt, "#999"), alpha=0.25,
                     label=TASK_TYPE_LABELS.get(tt, str(tt)))
               for tt in dict.fromkeys(tt for _, _, tt in segments)]
    return handles


def _legend_systems(ax, *, line: bool = True):
    handles = []
    for sysname in ("globalrep", "multirep"):
        if line:
            handles.append(Line2D([0], [0], color=SYSTEM_COLORS[sysname],
                                  ls=SYSTEM_LS[sysname], lw=_LW, label=SYSTEM_LABELS[sysname]))
        else:
            handles.append(Patch(facecolor="#bbbbbb", hatch=SYSTEM_HATCH[sysname],
                                 edgecolor="black", label=SYSTEM_LABELS[sysname]))
    ax.legend(handles=handles, title="System")


# =========================================================================
# 1. Accuracy / loss over rounds (globalrep vs multirep), per dataset
# =========================================================================

def plot_metric_over_rounds(pair: ExperimentPair, task_type: int, col: str,
                            ylabel: str, method: str = "mean",
                            task_indices: list[int] | None = None,
                            title_suffix: str = "") -> plt.Figure:
    """Compare a per-round metric (accuracy or loss) between systems for one dataset.

    *col* is a column in the global_accuracy table.  *task_indices*, if given,
    restricts to a subset of tasks (used for the by-third graphs).
    """
    dataset = TT_DATASET[task_type]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False
    for system, exp in pair.items():
        ga = exp.global_accuracy()
        if ga.empty or col not in ga.columns:
            continue
        sub = ga[ga["dataset"].str.lower() == dataset]
        if task_indices is not None:
            sub = sub[sub["task_index"].isin(task_indices)]
        sub = sub[sub["round"] <= ROUND_CAP[task_type]]
        if sub.empty:
            continue
        x, center, lo, hi = _curve(sub, col, method)
        color = SYSTEM_COLORS[system]
        ax.plot(x, center, color=color, ls=SYSTEM_LS[system], lw=_LW,
                label=f"{SYSTEM_LABELS[system]} ({_n_curves(sub)} distinct curves)")
        ax.fill_between(x, lo, hi, color=color, alpha=_FILL_ALPHA)
        plotted = True

    band = "±1 std" if method == "mean" else "IQR"
    ax.set_xlabel("FL round")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{TASK_TYPE_LABELS[task_type]} — {ylabel} over rounds "
        f"({method}, band={band}){title_suffix}"
    )
    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(title="System", fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_thirds(pair: ExperimentPair, task_type: int, col: str,
                       ylabel: str, method: str = "mean") -> plt.Figure:
    """1×3 panel: the same metric for the first / middle / last third of tasks."""
    dataset = TT_DATASET[task_type]
    indices = _dataset_indices(pair, dataset)
    buckets = _thirds(indices)
    names = ["First third", "Middle third", "Last third"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    band = "±1 std" if method == "mean" else "IQR"
    for ax, bucket, name in zip(axes, buckets, names):
        for system, exp in pair.items():
            ga = exp.global_accuracy()
            if ga.empty or col not in ga.columns:
                continue
            sub = ga[(ga["dataset"].str.lower() == dataset) & (ga["task_index"].isin(bucket))]
            sub = sub[sub["round"] <= ROUND_CAP[task_type]]
            if sub.empty:
                continue
            x, center, lo, hi = _curve(sub, col, method)
            color = SYSTEM_COLORS[system]
            ax.plot(x, center, color=color, ls=SYSTEM_LS[system], lw=_LW,
                    label=SYSTEM_LABELS[system])
            ax.fill_between(x, lo, hi, color=color, alpha=_FILL_ALPHA)
        rng = f"tasks {bucket[0]}–{bucket[-1]}" if bucket else "no tasks"
        ax.set_title(f"{name} ({rng})")
        ax.set_xlabel("FL round")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(title="System", fontsize=8)
    fig.suptitle(
        f"{TASK_TYPE_LABELS[task_type]} — {ylabel} over rounds, by session third "
        f"({method}, band={band})"
    )
    fig.tight_layout()
    return fig


# =========================================================================
# 2. Final accuracy & time-to-accuracy
# =========================================================================

def _final_accuracy(exp: ExperimentRuns) -> pd.DataFrame:
    """Per (run, task) last-round accuracy. Columns: dataset, accuracy."""
    ga = exp.global_accuracy()
    if ga.empty:
        return pd.DataFrame(columns=["dataset", "accuracy"])
    # one row per distinct experimental curve (collapse fingerprint cache clones)
    key = ["run", "fingerprint", "dataset"] if "fingerprint" in ga.columns else ["run", "task_index", "dataset"]
    last = (
        ga.sort_values("round")
        .groupby(key, sort=False)
        .last()
        .reset_index()
    )
    out = last[["dataset", "objective_global_accuracy"]].rename(
        columns={"objective_global_accuracy": "accuracy"}
    )
    out["dataset"] = out["dataset"].str.lower()
    return out


def plot_final_accuracy(pair: ExperimentPair) -> plt.Figure:
    """Grouped bar of mean final-round accuracy per dataset, globalrep vs multirep."""
    datasets = ["mnist", "cifar-10"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(datasets))

    for i, (system, exp) in enumerate(pair.items()):
        fa = _final_accuracy(exp)
        means, stds = [], []
        for ds in datasets:
            vals = fa.loc[fa["dataset"] == ds, "accuracy"]
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
        xpos = x - 0.4 + i * width + width / 2
        ax.bar(xpos, means, width, yerr=stds, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[MNIST_TT], TASK_TYPE_LABELS[CIFAR_TT]])
    ax.set_ylabel("Final-round global accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final accuracy by dataset (mean ±1 std across tasks·runs)")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _time_to_accuracy(exp: ExperimentRuns, mode: str) -> pd.DataFrame:
    """Rounds for each (run, task) to first reach the target. Columns: dataset, rounds."""
    ga = exp.global_accuracy()
    if ga.empty:
        return pd.DataFrame(columns=["dataset", "rounds"])
    rows = []
    key = ["run", "fingerprint", "dataset"] if "fingerprint" in ga.columns else ["run", "task_index", "dataset"]
    for _, g in ga.groupby(key, sort=False):
        ds = g["dataset"].iloc[0]
        g = g.sort_values("round")
        acc = g["objective_global_accuracy"]
        if mode == "fraction":
            target = ACC_FRACTION * acc.max()
        else:
            target = ACC_THRESHOLDS.get(ds.lower())
            if target is None:
                continue
        hit = g[acc >= target]
        if hit.empty:
            continue
        rows.append({"dataset": ds.lower(), "rounds": int(hit["round"].iloc[0])})
    return pd.DataFrame(rows, columns=["dataset", "rounds"]) if rows else pd.DataFrame(columns=["dataset", "rounds"])


def plot_time_to_accuracy(pair: ExperimentPair, mode: str = "threshold") -> plt.Figure:
    """Grouped bar of mean rounds-to-target per dataset, globalrep vs multirep.

    mode='threshold' uses fixed per-dataset targets; mode='fraction' uses 90%
    of each task's own final accuracy.
    """
    datasets = ["mnist", "cifar-10"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(datasets))

    for i, (system, exp) in enumerate(pair.items()):
        tt = _time_to_accuracy(exp, mode)
        means, stds = [], []
        for ds in datasets:
            vals = tt.loc[tt["dataset"] == ds, "rounds"]
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
        xpos = x - 0.4 + i * width + width / 2
        ax.bar(xpos, means, width, yerr=stds, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])

    if mode == "fraction":
        target_txt = f"{int(ACC_FRACTION * 100)}% of each task's final accuracy"
    else:
        target_txt = f"MNIST≥{ACC_THRESHOLDS['mnist']}, CIFAR-10≥{ACC_THRESHOLDS['cifar-10']}"
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[MNIST_TT], TASK_TYPE_LABELS[CIFAR_TT]])
    ax.set_ylabel("Rounds to reach target (lower = faster)")
    ax.set_title(f"Time-to-accuracy — target: {target_txt}")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 3. Participant selection rate per dataset, by behavior
# =========================================================================

def plot_selection_rate_by_behavior(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """Grouped bar: selection rate per behavior, globalrep vs multirep, one dataset.

    Only tasks of *task_type* are considered.  Bars are coloured by behavior;
    system is distinguished by hatch.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # behaviors actually present (preserve canonical order)
    present = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep.loc[rep["task_type"] == task_type, "behavior"].unique())
    behaviors = [b for b in BEHAVIOR_ORDER if b in present]
    if not behaviors:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(behaviors))

    for i, (system, exp) in enumerate(pair.items()):
        rep = exp.reputation_timeline()
        sub = rep[rep["task_type"] == task_type]
        rates = sub.groupby("behavior")["was_selected"].mean()
        vals = [rates.get(b, np.nan) for b in behaviors]
        xpos = x - 0.4 + i * width + width / 2
        ax.bar(xpos, vals, width,
               color=[BEHAVIOR_COLORS.get(b, "#888") for b in behaviors],
               hatch=SYSTEM_HATCH[system], edgecolor="black", linewidth=0.8, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Selection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]} participant selection rate by behavior")
    # legend: behavior colours + system hatch
    beh_handles = [Patch(facecolor=BEHAVIOR_COLORS.get(b, "#888"), edgecolor="black",
                         label=BEHAVIOR_LABELS.get(b, b)) for b in behaviors]
    sys_handles = [Patch(facecolor="white", hatch=SYSTEM_HATCH[s], edgecolor="black",
                        label=SYSTEM_LABELS[s]) for s in systems]
    leg1 = ax.legend(handles=beh_handles, title="Behavior", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=sys_handles, title="System", loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 4. TR / GIR development + selection rate over tasks
# =========================================================================

def plot_tr_development(pair: ExperimentPair) -> plt.Figure:
    """Mean TR per behavior over task index, one subplot per task type.

    Multi-rep keeps a *per-task-type* TR, so its MNIST and CIFAR lines diverge
    and go flat while the other dataset runs.  Global-rep keeps a *single global*
    task-reputation bucket shared across all task types, so the same line is drawn
    in both panels (it keeps climbing across dataset switches).  Behaviour →
    colour, system → line style.  Dataset runs are tinted; dashed lines mark
    switches.
    """
    task_types = [MNIST_TT, CIFAR_TT]
    fig, axes = plt.subplots(1, len(task_types), figsize=(16, 4.5), sharey=True)
    if len(task_types) == 1:
        axes = [axes]

    for ax, tt in zip(axes, task_types):
        ds_handles = _mark_dataset_switches(ax, pair)
        for system, exp in pair.items():
            rep = exp.reputation_timeline()
            if rep.empty:
                continue
            # Group by each user's behaviour *on this panel's dataset* (constant
            # per user), not the per-task behaviour.  For task-hoppers the per-task
            # label flips at every dataset switch, which would otherwise swap users
            # in/out of a behaviour line and produce a spurious rise/crash sawtooth.
            rep = rep.assign(_beh=_behavior_on_task_type(rep, tt))
            if exp.system == "globalrep":
                # single shared global bucket → use tr_post (the live bucket value)
                agg = rep.dropna(subset=["_beh"]).groupby(["_beh", "task_index"])["tr_post"].mean().reset_index()
                agg = agg.rename(columns={"tr_post": "_tr", "_beh": "behavior"})
            else:
                if "tr_all_post" not in rep.columns:
                    continue
                val = rep["tr_all_post"].apply(
                    lambda d: d.get(tt) if isinstance(d, dict) else None
                )
                agg = (rep.assign(_tr=val).dropna(subset=["_tr", "_beh"])
                       .groupby(["_beh", "task_index"])["_tr"].mean().reset_index()
                       .rename(columns={"_beh": "behavior"}))
            for behavior, grp in agg.groupby("behavior"):
                grp = grp.sort_values("task_index")
                ax.plot(grp["task_index"], grp["_tr"],
                        color=BEHAVIOR_COLORS.get(behavior, "#888"),
                        ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
        ax.set_title(f"{TASK_TYPE_LABELS[tt]} panel")
        ax.set_xlabel("Task index")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Task Reputation (TR)")
    _add_behavior_system_legend(axes[-1], pair, ds_handles)
    fig.suptitle("Task-reputation development  (colour=behavior, style=system; "
                 "global-rep = one shared bucket shown in both panels)")
    fig.tight_layout()
    return fig


def plot_gir_development(pair: ExperimentPair) -> plt.Figure:
    """Mean GIR per behavior over task index.

    GIR is a multi-rep–only concept: global-rep uses a single global task
    reputation and *no* integrity layer, so its GIR is identically zero by
    design.  Both systems are drawn (global-rep should sit flat at 0); a note
    flags any non-zero global-rep GIR as the pre-fix contamination that a
    re-run will clear.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ds_handles = _mark_dataset_switches(ax, pair)
    globalrep_nonzero = False
    for system, exp in pair.items():
        rep = exp.reputation_timeline()
        if rep.empty:
            continue
        if exp.system == "globalrep" and rep["gir_post"].abs().max() > _EPS:
            globalrep_nonzero = True
        agg = rep.groupby(["behavior", "task_index"])["gir_post"].mean().reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("task_index")
            ax.plot(grp["task_index"], grp["gir_post"],
                    color=BEHAVIOR_COLORS.get(behavior, "#888"),
                    ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Global Integrity Reputation (GIR)")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_title("GIR development (multi-rep only; global-rep has no GIR by design)")
    if globalrep_nonzero:
        ax.text(0.5, 0.94, "⚠ global-rep GIR ≠ 0 → pre-fix data; re-run to clear",
                ha="center", va="top", transform=ax.transAxes, fontsize=8, color="#b00")
    _add_behavior_system_legend(ax, pair, ds_handles)
    fig.tight_layout()
    return fig


def plot_selection_rate_over_time(pair: ExperimentPair) -> plt.Figure:
    """Mean selection rate per behavior over task index. Behaviour→colour, system→style."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ds_handles = _mark_dataset_switches(ax, pair)
    for system, exp in pair.items():
        rep = exp.reputation_timeline()
        if rep.empty:
            continue
        agg = rep.groupby(["behavior", "task_index"])["was_selected"].mean().reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("task_index")
            ax.plot(grp["task_index"], grp["was_selected"],
                    color=BEHAVIOR_COLORS.get(behavior, "#888"),
                    ls=SYSTEM_LS[system], lw=_LW, alpha=0.8)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Selection rate (fraction of users)")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_title("Selection rate over tasks (colour=behavior, style=system)")
    _add_behavior_system_legend(ax, pair, ds_handles)
    fig.tight_layout()
    return fig


# =========================================================================
# 4b. Single-dataset progression (only MNIST tasks, or only CIFAR tasks)
# =========================================================================

def _dataset_task_order(pair: ExperimentPair, task_type: int) -> dict[int, int]:
    """task_index -> consecutive 0-based position among tasks of *task_type*."""
    mp = _task_dataset_map(pair)
    tis = sorted(ti for ti, tt in mp.items() if tt == task_type)
    return {ti: i for i, ti in enumerate(tis)}


def _progression_value(sub: pd.DataFrame, exp: ExperimentRuns, metric: str, task_type: int):
    if metric == "tr":
        if exp.system == "globalrep":
            return sub["tr_post"]
        return sub["tr_all_post"].apply(lambda d: d.get(task_type) if isinstance(d, dict) else None)
    if metric == "gir":
        return sub["gir_post"]
    return sub["was_selected"].astype(float)


def plot_metric_progression(pair: ExperimentPair, task_type: int, metric: str) -> plt.Figure:
    """A metric over *only* the tasks of one dataset, x compressed to consecutive
    dataset-task number — no flat gaps from the other dataset.

    multi-rep shows that dataset's per-task reputation building continuously;
    global-rep shows its single global bucket sampled at those tasks (so it keeps
    rising across both datasets — the same global line appears in both variants).
    """
    order = _dataset_task_order(pair, task_type)
    ylabel = {"tr": "Task Reputation (TR)", "gir": "Global Integrity Reputation (GIR)",
              "selection": "Selection rate"}[metric]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for system, exp in pair.items():
        rep = exp.reputation_timeline()
        if rep.empty:
            continue
        sub = rep[rep["task_index"].isin(order)].copy()
        if sub.empty:
            continue
        sub["_v"] = _progression_value(sub, exp, metric, task_type)
        sub["_x"] = sub["task_index"].map(order)
        agg = sub.dropna(subset=["_v"]).groupby(["behavior", "_x"])["_v"].mean().reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("_x")
            ax.plot(grp["_x"], grp["_v"], color=BEHAVIOR_COLORS.get(behavior, "#888"),
                    ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
    ax.set_xlabel(f"{TASK_TYPE_LABELS[task_type]} task # (consecutive, other dataset removed)")
    ax.set_ylabel(ylabel)
    if metric != "selection":
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]}-only {ylabel} progression "
                 "(colour=behavior, style=system)")
    _add_behavior_system_legend(ax, pair)
    fig.tight_layout()
    return fig


# =========================================================================
# 5. Cold-start: do GIR-only users get picked for CIFAR before no-rep users?
# =========================================================================

def plot_cold_start_selection(pair: ExperimentPair) -> plt.Figure:
    """Selection rate on CIFAR tasks, bucketed by pre-task reputation state.

    Among users with *no* CIFAR task-reputation, compare those who carry global
    integrity reputation ('GIR, no CIFAR-TR') against those with none
    ('No GIR, no CIFAR-TR').  If multirep is working, GIR alone should *not*
    buy CIFAR selection the way it does under globalrep.
    """
    bucket_names = ["GIR, no CIFAR-TR", "No GIR, no CIFAR-TR"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(bucket_names))

    for i, (system, exp) in enumerate(pair.items()):
        rep = exp.reputation_timeline()
        sub = rep[rep["task_type"] == CIFAR_TT].copy()
        if sub.empty:
            continue
        sub["cifar_tr_pre"] = sub["tr_all_pre"].apply(
            lambda d: d.get(CIFAR_TT, 0.0) if isinstance(d, dict) else 0.0
        )
        no_tr = sub[sub["cifar_tr_pre"] <= _EPS]
        gir_only = no_tr[no_tr["gir_pre"] > _EPS]
        cold = no_tr[no_tr["gir_pre"] <= _EPS]
        vals = [
            gir_only["was_selected"].mean() if len(gir_only) else np.nan,
            cold["was_selected"].mean() if len(cold) else np.nan,
        ]
        xpos = x - 0.4 + i * width + width / 2
        ax.bar(xpos, vals, width, color=SYSTEM_COLORS[system],
               edgecolor="black", linewidth=0.7, alpha=0.85, label=SYSTEM_LABELS[system])

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names)
    ax.set_ylabel("Selection rate on CIFAR-10 tasks")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cold-start: does global reputation buy CIFAR selection\n"
                 "without any CIFAR task reputation?")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 6. Participants getting kicked
# =========================================================================

def _kicked_records(exp: ExperimentRuns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kicks, participations).

    kicks         : one row per (run, task, user) disqualification —
                    columns run, task_index, user, behavior, round_kicked.
    participations: distinct (run, task, user) seen per behavior — for rates.
    """
    kicks, parts = [], []
    for run, ti, _ds, _tt, u in exp.iter_task_users():
        def _beh(v):
            return v.name.lower() if hasattr(v, "name") else str(v).lower()

        for un, grp in u.groupby("user_number"):
            role = grp["role"].iloc[0] if "role" in grp.columns else grp["behavior"].iloc[0]
            parts.append({"run": run, "task_index": ti, "user": un, "behavior": _beh(role)})
            disq = grp[grp["state"] == "disqualified"]
            if not disq.empty:
                kicks.append({
                    "run": run, "task_index": ti, "user": un,
                    "behavior": _beh(role),
                    "round_kicked": int(disq["round"].min()),
                })
    return pd.DataFrame(kicks), pd.DataFrame(parts)


def plot_kicked_round(pair: ExperimentPair) -> plt.Figure:
    """Mean within-task round at which each behavior is disqualified (min/max bars)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    present = set()
    data = {}
    for system, exp in pair.items():
        kicks, _ = _kicked_records(exp)
        data[system] = kicks
        if not kicks.empty:
            present |= set(kicks["behavior"].unique())
    behaviors = [b for b in BEHAVIOR_ORDER if b in present]
    if not behaviors:
        ax.text(0.5, 0.5, "No disqualifications", ha="center", va="center", transform=ax.transAxes)
        return fig

    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(behaviors))
    for i, system in enumerate(systems):
        kicks = data[system]
        means, lo, hi = [], [], []
        for b in behaviors:
            v = kicks.loc[kicks["behavior"] == b, "round_kicked"] if not kicks.empty else pd.Series(dtype=float)
            if len(v):
                m = v.mean(); means.append(m); lo.append(m - v.min()); hi.append(v.max() - m)
            else:
                means.append(np.nan); lo.append(0); hi.append(0)
        xpos = x - 0.4 + i * width + width / 2
        ax.bar(xpos, means, width, yerr=[lo, hi], capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Round kicked within task (lower = sooner)")
    ax.set_title("When are participants disqualified? (bars = min/max across tasks·runs)")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_kicked_rate(pair: ExperimentPair) -> plt.Figure:
    """Fraction of task-participations that ended in disqualification, by behavior."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    present = set()
    rates = {}
    for system, exp in pair.items():
        kicks, parts = _kicked_records(exp)
        if parts.empty:
            rates[system] = {}
            continue
        present |= set(parts["behavior"].unique())
        n_part = parts.groupby("behavior").size()
        n_kick = kicks.groupby("behavior").size() if not kicks.empty else pd.Series(dtype=int)
        rates[system] = {b: (n_kick.get(b, 0) / n_part.get(b, 1)) for b in n_part.index}
    behaviors = [b for b in BEHAVIOR_ORDER if b in present]
    if not behaviors:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(behaviors))
    for i, system in enumerate(systems):
        vals = [rates[system].get(b, np.nan) for b in behaviors]
        xpos = x - 0.4 + i * width + width / 2
        ax.bar(xpos, vals, width, color=SYSTEM_COLORS[system],
               edgecolor="black", linewidth=0.7, alpha=0.85, label=SYSTEM_LABELS[system])
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Disqualification rate (kicks / participations)")
    ax.set_ylim(0, 1.05)
    ax.set_title("How often is each behavior disqualified?")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 7. Q-value special comparison (with-q vs without-q)
# =========================================================================

def _selection_waits(exp: ExperimentRuns) -> pd.DataFrame:
    """Per (run, user): max consecutive non-selected tasks and mean gap between
    selections.  Larger waits = a user is left idle longer."""
    rep = exp.reputation_timeline()
    rows = []
    if rep.empty:
        return pd.DataFrame(columns=["max_wait", "mean_gap"])
    for (_run, guid), grp in rep.groupby(["run", "guid"]):
        grp = grp.sort_values("task_index")
        sel = grp["was_selected"].to_numpy()
        # max run length of consecutive False
        max_wait = cur = 0
        for s in sel:
            cur = 0 if s else cur + 1
            max_wait = max(max_wait, cur)
        sel_pos = np.where(sel)[0]
        mean_gap = float(np.mean(np.diff(sel_pos))) if len(sel_pos) >= 2 else np.nan
        rows.append({"max_wait": max_wait, "mean_gap": mean_gap})
    return pd.DataFrame(rows)


def _qvalue_user_stats(exp: ExperimentRuns) -> pd.DataFrame:
    """Per (run, user): selection count, longest idle streak, mean TR."""
    rep = exp.reputation_timeline()
    rows = []
    if rep.empty:
        return pd.DataFrame(columns=["n_sel", "max_idle", "mean_tr"])
    for (_run, _guid), g in rep.groupby(["run", "guid"]):
        g = g.sort_values("task_index")
        sel = g["was_selected"].to_numpy().astype(bool)
        mw = cur = 0
        for s in sel:
            cur = 0 if s else cur + 1
            mw = max(mw, cur)
        rows.append({"n_sel": int(sel.sum()), "max_idle": mw,
                     "mean_tr": float(g["tr_post"].mean())})
    return pd.DataFrame(rows)


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def plot_qvalue_effect(with_q: ExperimentRuns, without_q: ExperimentRuns) -> plt.Figure:
    """Three clear views of what the Q-value (long-unselected bonus) does to
    selection fairness:

      (a) Lorenz curve of selection counts — how (un)equally picks are shared.
      (b) Per-user selection count, ranked — flat = everyone gets turns.
      (c) Selection count vs mean TR — without Q, picks track TR (rich-get-richer);
          the Q-value decouples them so idle good users still get in.
    """
    variants = [("With Q-value", "#1b9e77", _qvalue_user_stats(with_q)),
                ("Without Q-value", "#d95f02", _qvalue_user_stats(without_q))]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    for label, color, st in variants:
        if st.empty:
            continue
        # (a) Lorenz curve
        x = np.sort(st["n_sel"].to_numpy(dtype=float))
        cum = np.cumsum(x)
        cum = np.insert(cum / cum[-1], 0, 0) if cum[-1] > 0 else np.linspace(0, 1, len(x) + 1)
        frac = np.linspace(0, 1, len(cum))
        ax1.plot(frac, cum, color=color, lw=_LW, label=f"{label} (Gini={_gini(x):.2f})")
        # (b) ranked selection counts
        ranked = np.sort(st["n_sel"].to_numpy())[::-1]
        ax2.plot(range(1, len(ranked) + 1), ranked, color=color, lw=_LW, marker="o",
                 ms=3, label=label)
        # (c) selection count vs mean TR
        ax3.scatter(st["mean_tr"], st["n_sel"], color=color, alpha=0.6, s=24, label=label)

    ax1.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="perfect equality")
    ax1.set_xlabel("Cumulative share of users (poorest→richest)")
    ax1.set_ylabel("Cumulative share of selections")
    ax1.set_title("Selection inequality (Lorenz)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("User rank")
    ax2.set_ylabel("Times selected")
    ax2.set_title("Selection count per user, ranked")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("User mean TR")
    ax3.set_ylabel("Times selected")
    ax3.set_title("Selection vs reputation")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    fig.suptitle("Effect of the Q-value on selection fairness")
    fig.tight_layout()
    return fig


def plot_qvalue_selection_wait(with_q: ExperimentRuns, without_q: ExperimentRuns) -> plt.Figure:
    """Compare how long participants wait to be selected, with vs without the Q-value.

    Left: distribution of each user's *longest* idle streak (consecutive tasks
    not selected).  Right: mean gap between successive selections.  The Q-value
    is the long-unselected bonus, so disabling it should lengthen waits.
    """
    labels = ["With Q-value", "Without Q-value"]
    colors = ["#1b9e77", "#d95f02"]
    data = [_selection_waits(with_q), _selection_waits(without_q)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    mw = [d["max_wait"].dropna().to_numpy() for d in data]
    bp1 = ax1.boxplot(mw, patch_artist=True, tick_labels=labels)
    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax1.set_ylabel("Longest idle streak (consecutive tasks unselected)")
    ax1.set_title("Worst-case selection wait per user")
    ax1.grid(True, axis="y", alpha=0.3)

    mg = [d["mean_gap"].dropna().to_numpy() for d in data]
    bp2 = ax2.boxplot(mg, patch_artist=True, tick_labels=labels)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_ylabel("Mean gap between selections (tasks)")
    ax2.set_title("Typical selection gap per user")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Effect of the Q-value (selection-pressure bonus) on participant wait time")
    fig.tight_layout()
    return fig


def _coverage_curve(exp: ExperimentRuns) -> pd.DataFrame:
    """Per task_index (mean over runs): fraction of users selected at least once
    so far (coverage) and the Gini of cumulative selection counts."""
    rep = exp.reputation_timeline()
    if rep.empty:
        return pd.DataFrame(columns=["task_index", "coverage", "gini"])
    run_frames = []
    for _run, g in rep.groupby("run"):
        n_users = g["guid"].nunique()
        order = sorted(g["task_index"].unique())
        seen: set = set()
        counts: dict = {}
        cov, gini = [], []
        for ti in order:
            picked = g[(g["task_index"] == ti) & (g["was_selected"])]["guid"]
            for gid in picked:
                seen.add(gid)
                counts[gid] = counts.get(gid, 0) + 1
            cov.append(len(seen) / n_users if n_users else np.nan)
            vec = np.array(list(counts.values()) + [0] * (n_users - len(counts)), dtype=float)
            gini.append(_gini(vec))
        run_frames.append(pd.DataFrame({"task_index": order, "coverage": cov, "gini": gini}))
    allruns = pd.concat(run_frames, ignore_index=True)
    return allruns.groupby("task_index")[["coverage", "gini"]].mean().reset_index()


def plot_qvalue_coverage(with_q: ExperimentRuns, without_q: ExperimentRuns) -> plt.Figure:
    """How evenly participation is shared *over time*, with vs without the Q-value.

    Left: participation coverage — fraction of users selected at least once by
    task t.  With the Q-value, every participant gets a turn sooner, so coverage
    climbs to 1.0 faster.  Right: running Gini of cumulative selection counts —
    lower = picks shared more equally as the session goes on."""
    variants = [("With Q-value", "#1b9e77", _coverage_curve(with_q)),
                ("Without Q-value", "#d95f02", _coverage_curve(without_q))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color, c in variants:
        if c.empty:
            continue
        ax1.plot(c["task_index"], c["coverage"], color=color, lw=_LW, marker="o", ms=3, label=label)
        ax2.plot(c["task_index"], c["gini"], color=color, lw=_LW, marker="o", ms=3, label=label)
    ax1.set_xlabel("Task index"); ax1.set_ylabel("Fraction of users selected ≥ once")
    ax1.set_ylim(0, 1.05); ax1.set_title("Participation coverage over time")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Task index"); ax2.set_ylabel("Gini of cumulative selections")
    ax2.set_ylim(0, 1.0); ax2.set_title("Selection inequality over time")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.suptitle("Effect of the Q-value: how fast everyone gets a turn")
    fig.tight_layout()
    return fig


def _idle_streak_table(exp: ExperimentRuns) -> pd.DataFrame:
    """One row per (run, user, task): the idle streak *entering* this task
    (consecutive prior tasks the user was not selected for), the Q-value used for
    the selection decision (q_pre), and whether they were then selected."""
    rep = exp.reputation_timeline()
    if rep.empty:
        return pd.DataFrame(columns=["streak", "q_pre", "was_selected"])
    rep = rep.sort_values(["run", "guid", "task_index"])
    rows = []
    for (_run, _gid), g in rep.groupby(["run", "guid"]):
        streak = 0
        for _, row in g.iterrows():
            rows.append({"streak": streak, "q_pre": row["q_pre"],
                         "was_selected": bool(row["was_selected"])})
            streak = 0 if row["was_selected"] else streak + 1
    return pd.DataFrame(rows)


def plot_qvalue_mechanism(with_q: ExperimentRuns, without_q: ExperimentRuns,
                          max_streak: int = 8) -> plt.Figure:
    """The Q-value mechanism, made explicit.

    Left: probability of being selected vs how long a user has sat idle.  The
    Q-value is the long-unselected bonus, so with it P(selected) *rises* with the
    idle streak (idle users get pulled back in); without it, an unselected user
    has no recovery path and their chance stays low.  Right: the resulting idle-
    streak distribution — with the Q-value, task-opportunities pile up at short
    streaks; without it a long starvation tail appears."""
    variants = [("With Q-value", "#1b9e77", _idle_streak_table(with_q)),
                ("Without Q-value", "#d95f02", _idle_streak_table(without_q))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color, t in variants:
        if t.empty:
            continue
        t = t.copy()
        t["streak_c"] = t["streak"].clip(upper=max_streak)
        # (left) P(selected) vs idle streak
        p = t.groupby("streak_c")["was_selected"].mean().reset_index()
        ax1.plot(p["streak_c"], p["was_selected"], color=color, lw=_LW, marker="o", ms=4, label=label)
        # (right) distribution of idle streaks (share of task-opportunities)
        d = t.groupby("streak_c").size()
        d = (d / d.sum()).reset_index(name="frac")
        ax2.plot(d["streak_c"], d["frac"], color=color, lw=_LW, marker="o", ms=4, label=label)
    xlbl = f"Idle streak entering task (consecutive tasks unselected, capped at {max_streak})"
    ax1.set_xlabel(xlbl); ax1.set_ylabel("P(selected)")
    ax1.set_ylim(0, 1.05); ax1.set_title("Idle users get pulled back in")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel(xlbl); ax2.set_ylabel("Share of task-opportunities")
    ax2.set_title("Idle-streak distribution (Q prevents starvation tail)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.suptitle("Effect of the Q-value: the long-unselected bonus and its consequence")
    fig.tight_layout()
    return fig


# =========================================================================
# internal
# =========================================================================

def _add_behavior_system_legend(ax, pair: ExperimentPair, ds_handles: list | None = None) -> None:
    """Behaviour colours + system line styles, plus an optional dataset-tint key.

    *ds_handles* are the dataset Patch handles returned by ``_mark_dataset_switches``;
    pass them so the background tint (which dataset each task ran) is explained.
    """
    present = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep["behavior"].unique())
    behaviors = [b for b in BEHAVIOR_ORDER if b in present]
    beh_handles = [Line2D([0], [0], color=BEHAVIOR_COLORS.get(b, "#888"), lw=_LW,
                          label=BEHAVIOR_LABELS.get(b, b)) for b in behaviors]
    sys_handles = [Line2D([0], [0], color="#555", ls=SYSTEM_LS[s], lw=_LW,
                          label=SYSTEM_LABELS[s]) for s, _ in pair.items()]
    leg1 = ax.legend(handles=beh_handles, title="Behavior", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=sys_handles, title="System", loc="lower right", fontsize=8)
    if ds_handles:
        ax.add_artist(leg2)
        ax.legend(handles=ds_handles, title="Dataset (tint)", loc="lower left", fontsize=7)
