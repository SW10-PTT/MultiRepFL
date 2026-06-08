"""Graphs that group participants by their data-split category *and* behavior.

Data-split category is read from each user's name:
  * avg / mixed-distribution presets → "MNIST-heavy" / "CIFAR-heavy" / "Balanced"
  * task-hopper presets             → "Both" / "MNIST-only" / "CIFAR-only"

Every plot compares global-rep vs multi-rep.  Behaviour keeps its canonical
colour; the data-split category is the x-axis (bar charts) or the subplot facet
(development line charts); the system is a bar group / line style.
"""

from __future__ import annotations

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
)
from analysis.multirep_aggregate_plots import (
    BEHAVIOR_COLORS,
    BEHAVIOR_LABELS,
    BEHAVIOR_ORDER,
    SYSTEM_LABELS,
    SYSTEM_LS,
    _mark_dataset_switches,
)

_LW = 2
_EPS = 1e-6
_BALANCE_INITIAL = 100.0

# Preferred display order; any other category found is appended.
# Three naming schemes appear across the preset families:
#   avg-distribution   → "MNIST-heavy" / "Balanced" / "CIFAR-heavy"
#   mixed-distribution → "MNIST-strong" / "Average" / "CIFAR-strong"
#   task-hopper        → "MNIST-only" / "Both" / "CIFAR-only"
_CATEGORY_ORDER = [
    "MNIST-heavy", "Balanced", "CIFAR-heavy",
    "MNIST-strong", "Average", "CIFAR-strong",
    "MNIST-only", "Both", "CIFAR-only",
]


def split_category(name: str) -> str:
    """Map a participant name to its data-split category (MNIST-leaning,
    balanced, or CIFAR-leaning), across all preset naming schemes."""
    nl = (name or "").lower()
    # avg-distribution / mixed-distribution explicit tags
    if "mnist-heavy" in nl:
        return "MNIST-heavy"
    if "cifar-heavy" in nl:
        return "CIFAR-heavy"
    if "balanced" in nl:
        return "Balanced"
    if "mnist-strong" in nl:
        return "MNIST-strong"
    if "cifar-strong" in nl:
        return "CIFAR-strong"
    if "average" in nl:
        return "Average"
    # task-hopper style: descriptor after the id token (e.g. "H3 Both", "F2 MNIST")
    rest = name.split(None, 1)[1].strip().lower() if name and " " in name else nl
    if rest.startswith("both"):
        return "Both"
    if rest.startswith("mnist"):
        return "MNIST-only"
    if rest.startswith("cifar"):
        return "CIFAR-only"
    return "Other"


def split_dataset_bias(category: str) -> int | None:
    """The task_type a category is data-rich in, or None if balanced/unknown.
    Used to test 'strong-on-X selected more for X'."""
    if category in ("MNIST-heavy", "MNIST-strong", "MNIST-only"):
        return MNIST_TT
    if category in ("CIFAR-heavy", "CIFAR-strong", "CIFAR-only"):
        return CIFAR_TT
    return None


def _with_category(rep: pd.DataFrame) -> pd.DataFrame:
    rep = rep.copy()
    rep["split"] = rep["user_name"].map(split_category)
    return rep


def _present_categories(pair: ExperimentPair) -> list[str]:
    present = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep["user_name"].map(split_category).unique())
    ordered = [c for c in _CATEGORY_ORDER if c in present]
    ordered += [c for c in sorted(present) if c not in ordered]
    return ordered


def _present_behaviors(pair: ExperimentPair) -> list[str]:
    present = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep["behavior"].unique())
    return [b for b in BEHAVIOR_ORDER if b in present]


# =========================================================================
# 1. Selection rate by (data-split, behavior), per dataset
# =========================================================================

def plot_selection_rate_by_split(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """Selection rate per (data-split category, behavior); one subplot per system."""
    cats = _present_categories(pair)
    behaviors = _present_behaviors(pair)
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7 * len(systems), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = _with_category(exp.reputation_timeline())
        sub = rep[rep["task_type"] == task_type]
        x = np.arange(len(cats))
        width = 0.8 / max(1, len(behaviors))
        for j, b in enumerate(behaviors):
            rates = [
                sub[(sub["split"] == c) & (sub["behavior"] == b)]["was_selected"].mean()
                for c in cats
            ]
            ax.bar(x - 0.4 + j * width + width / 2, rates, width,
                   color=BEHAVIOR_COLORS.get(b, "#888"), edgecolor="black",
                   linewidth=0.6, label=BEHAVIOR_LABELS.get(b, b))
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Selection rate")
    axes[-1].legend(title="Behavior", fontsize=8)
    fig.suptitle(f"{TASK_TYPE_LABELS[task_type]} selection rate by data-split & behavior")
    fig.text(0.5, 0.01,
             "Realised rate ≈ N/total for everyone — the Q-value rotates the fixed picks over the "
             "session. See selection-propensity (score) for the merit signal.",
             ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# =========================================================================
# 1b. Selection PROPENSITY (merit score) by data-split, MNIST vs CIFAR
# =========================================================================

def plot_selection_propensity_by_split(pair: ExperimentPair) -> plt.Figure:
    """Mean selection *score* per data-split category, split by dataset.

    Realised selection rate is ~flat (the Q-value rotates the fixed N picks
    across everyone over a long session), so it hides preference.  The selection
    *score* is the merit signal the protocol ranks on, and it should be higher
    for a participant on the dataset they hold strong data for.  Each category's
    data-rich dataset is marked with ★.
    """
    cats = _present_categories(pair)
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7 * len(systems), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ds_types = [MNIST_TT, CIFAR_TT]
    ds_colors = {MNIST_TT: "#2196F3", CIFAR_TT: "#FF9800"}

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = _with_category(exp.reputation_timeline())
        x = np.arange(len(cats))
        width = 0.8 / len(ds_types)
        for j, tt in enumerate(ds_types):
            vals = [rep[(rep["split"] == c) & (rep["task_type"] == tt)]["selection_score"].mean()
                    for c in cats]
            ax.bar(x - 0.4 + j * width + width / 2, vals, width,
                   color=ds_colors[tt], edgecolor="black", linewidth=0.6,
                   label=TASK_TYPE_LABELS[tt])
        # ★ over the data-rich dataset bar for each category
        for xi, c in zip(x, cats):
            bias = split_dataset_bias(c)
            if bias in ds_types:
                j = ds_types.index(bias)
                ax.annotate("★", (xi - 0.4 + j * width + width / 2, 0),
                            ha="center", va="bottom", color="#444", fontsize=12)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean selection score (merit)")
    axes[-1].legend(title="Task dataset", fontsize=8)
    fig.suptitle("Selection propensity by data-split & dataset "
                 "(★ = category's data-rich dataset; higher there = correct preference)")
    fig.tight_layout()
    return fig


# =========================================================================
# 2. Final balance by (data-split, behavior)
# =========================================================================

def _per_task_delta(rep: pd.DataFrame) -> pd.DataFrame:
    """Add ``_delta`` = net ETH change for each task (balance_post - balance_pre).

    Summing deltas is additive and attributable, unlike the cumulative balance:
    a user who free-rides on one dataset and is honest on the other has each
    task's gain/loss credited to that task's *own* behavior and dataset, with no
    double-counting.
    """
    rep = rep.copy()
    rep["_delta"] = rep["balance_post"] - rep["balance_pre"]
    return rep


def plot_net_earnings_by_split(pair: ExperimentPair) -> plt.Figure:
    """Mean net ETH earned per participant, grouped by (data-split, behavior).

    Earnings are summed per-task deltas (per user, per behavior), so a
    mixed-behavior user contributes their free-rider-dataset tasks to the
    free-rider bar and their honest-dataset tasks to the honest bar — their two
    parts sum to their true total with no double count.
    """
    cats = _present_categories(pair)
    behaviors = _present_behaviors(pair)
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7 * len(systems), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = _per_task_delta(_with_category(exp.reputation_timeline()))
        # net earnings per (run, user, behavior) = sum of that behavior's task deltas
        per_user = (rep.groupby(["run", "user_name", "split", "behavior"])["_delta"]
                    .sum().reset_index())
        x = np.arange(len(cats))
        width = 0.8 / max(1, len(behaviors))
        for j, b in enumerate(behaviors):
            means, errs = [], []
            for c in cats:
                v = per_user[(per_user["split"] == c) & (per_user["behavior"] == b)]["_delta"]
                means.append(v.mean() if len(v) else np.nan)
                errs.append(v.std() if len(v) > 1 else 0.0)
            ax.bar(x - 0.4 + j * width + width / 2, means, width, yerr=errs, capsize=3,
                   color=BEHAVIOR_COLORS.get(b, "#888"), edgecolor="black",
                   linewidth=0.6, label=BEHAVIOR_LABELS.get(b, b))
        ax.axhline(0, color="#555", lw=1)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Net earnings (ETH, summed task deltas)")
    axes[-1].legend(title="Behavior", fontsize=8)
    fig.suptitle("Net earnings by data-split & behavior — both datasets combined "
                 "(0 = break-even; see *_by_dataset for the MNIST/CIFAR split)")
    fig.tight_layout()
    return fig


def plot_net_earnings_by_split_dataset(pair: ExperimentPair) -> plt.Figure:
    """Net ETH earned per data-split category, split by dataset (MNIST vs CIFAR).

    Even all-honest users differ: a participant data-rich in one dataset should
    earn more there.  Earnings are summed per-task deltas attributed to each
    task's dataset, so the per-dataset asymmetry is visible.
    """
    cats = _present_categories(pair)
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7 * len(systems), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ds_types = [MNIST_TT, CIFAR_TT]
    ds_colors = {MNIST_TT: "#2196F3", CIFAR_TT: "#FF9800"}

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = _per_task_delta(_with_category(exp.reputation_timeline()))
        per_user = (rep.groupby(["run", "user_name", "split", "task_type"])["_delta"]
                    .sum().reset_index())
        x = np.arange(len(cats))
        width = 0.8 / len(ds_types)
        for j, tt in enumerate(ds_types):
            means, errs = [], []
            for c in cats:
                v = per_user[(per_user["split"] == c) & (per_user["task_type"] == tt)]["_delta"]
                means.append(v.mean() if len(v) else np.nan)
                errs.append(v.std() if len(v) > 1 else 0.0)
            ax.bar(x - 0.4 + j * width + width / 2, means, width, yerr=errs, capsize=3,
                   color=ds_colors[tt], edgecolor="black", linewidth=0.6,
                   label=TASK_TYPE_LABELS[tt])
        for xi, c in zip(x, cats):
            bias = split_dataset_bias(c)
            if bias in ds_types:
                ax.annotate("★", (xi - 0.4 + ds_types.index(bias) * width + width / 2, 0),
                            ha="center", va="bottom", color="#444", fontsize=12)
        ax.axhline(0, color="#555", lw=1)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Net earnings (ETH, summed task deltas)")
    axes[-1].legend(title="Task dataset", fontsize=8)
    fig.suptitle("Net earnings by data-split, per dataset "
                 "(★ = category's data-rich dataset; earn more there = data-aware reward)")
    fig.tight_layout()
    return fig


# =========================================================================
# 2c. Mixed-behavior users — per-dataset profile
# =========================================================================

def _mixed_user_table(exp) -> pd.DataFrame:
    """Per (user, task_type): mean net earnings across runs + the user's behavior
    on that dataset. Restricted to users whose behavior differs across datasets."""
    rep = _per_task_delta(exp.reputation_timeline())
    per_run = (rep.groupby(["run", "user_name", "task_type"])
               .agg(delta=("_delta", "sum"), behavior=("behavior", "first"))
               .reset_index())
    agg = (per_run.groupby(["user_name", "task_type"])
           .agg(delta=("delta", "mean"), behavior=("behavior", "first"))
           .reset_index())
    nbeh = agg.groupby("user_name")["behavior"].nunique()
    mixed = nbeh[nbeh > 1].index
    return agg[agg["user_name"].isin(mixed)]


def plot_mixed_behavior_users(pair: ExperimentPair) -> plt.Figure:
    """Per-dataset net earnings for users whose behavior differs across datasets
    (e.g. free-rider on MNIST, honest on CIFAR).  Each user gets one bar per
    dataset, coloured by the behavior they hold on that dataset — showing the
    same identity treated correctly per dataset.
    """
    systems = [s for s, _ in pair.items()]
    tables = {s: _mixed_user_table(exp) for s, exp in pair.items()}
    users = sorted(set().union(*[set(t["user_name"]) for t in tables.values()]))
    fig, axes = plt.subplots(1, len(systems), figsize=(max(7, len(users) * 0.9) * len(systems) / 2 + 2, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    if not users:
        for ax in axes:
            ax.text(0.5, 0.5, "No behavior-mixed users in this experiment",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
        fig.suptitle("Mixed-behavior users (none present)")
        fig.tight_layout()
        return fig

    ds_types = [MNIST_TT, CIFAR_TT]
    x = np.arange(len(users))
    width = 0.8 / len(ds_types)
    present_beh = set()
    for ax, system in zip(axes, systems):
        t = tables[system].set_index(["user_name", "task_type"])
        for j, tt in enumerate(ds_types):
            for xi, u in zip(x, users):
                if (u, tt) not in t.index:
                    continue
                row = t.loc[(u, tt)]
                beh = row["behavior"]
                present_beh.add(beh)
                ax.bar(xi - 0.4 + j * width + width / 2, row["delta"], width,
                       color=BEHAVIOR_COLORS.get(beh, "#888"), edgecolor="black", linewidth=0.6)
                ax.annotate(TASK_TYPE_LABELS[tt][0], (xi - 0.4 + j * width + width / 2, 0),
                            ha="center", va="bottom", fontsize=7, color="#333")
        ax.axhline(0, color="#555", lw=1)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xticks(x)
        ax.set_xticklabels(users, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Net earnings (ETH)")
    handles = [Patch(facecolor=BEHAVIOR_COLORS.get(b, "#888"), edgecolor="black",
                     label=BEHAVIOR_LABELS.get(b, b)) for b in BEHAVIOR_ORDER if b in present_beh]
    axes[-1].legend(handles=handles, title="Behavior on dataset", fontsize=8)
    fig.suptitle("Mixed-behavior users: per-dataset earnings (bar letter = M[NIST]/C[IFAR]; "
                 "colour = behavior on that dataset)")
    fig.tight_layout()
    return fig


# =========================================================================
# 3. TR / GIR development faceted by data-split category
# =========================================================================

def _facet_development(pair: ExperimentPair, value_fn, ylabel: str, title: str,
                       task_type: int | None = None) -> plt.Figure:
    cats = _present_categories(pair)
    fig, axes = plt.subplots(1, len(cats), figsize=(5.2 * len(cats), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, cat in zip(axes, cats):
        _mark_dataset_switches(ax, pair)
        for system, exp in pair.items():
            rep = _with_category(exp.reputation_timeline())
            rep = rep[rep["split"] == cat]
            if rep.empty:
                continue
            rep = rep.assign(_v=value_fn(rep, exp, task_type))
            agg = rep.dropna(subset=["_v"]).groupby(["behavior", "task_index"])["_v"].mean().reset_index()
            for behavior, grp in agg.groupby("behavior"):
                grp = grp.sort_values("task_index")
                ax.plot(grp["task_index"], grp["_v"],
                        color=BEHAVIOR_COLORS.get(behavior, "#888"),
                        ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
        ax.set_title(cat)
        ax.set_xlabel("Task index")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    _split_legend(axes[-1], pair)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _tr_value(rep: pd.DataFrame, exp, task_type: int | None):
    """TR series: global-rep uses the single shared bucket (tr_post); multi-rep
    uses the per-task-type value for *task_type*."""
    if exp.system == "globalrep" or task_type is None:
        return rep["tr_post"]
    return rep["tr_all_post"].apply(lambda d: d.get(task_type) if isinstance(d, dict) else None)


def plot_tr_by_split(pair: ExperimentPair, task_type: int) -> plt.Figure:
    return _facet_development(
        pair, _tr_value, "Task Reputation (TR)",
        f"{TASK_TYPE_LABELS[task_type]} task-reputation by data-split "
        f"(colour=behavior, style=system)",
        task_type=task_type,
    )


def plot_gir_by_split(pair: ExperimentPair) -> plt.Figure:
    return _facet_development(
        pair, lambda rep, exp, tt: rep["gir_post"], "Global Integrity Reputation (GIR)",
        "GIR by data-split (multi-rep only; global-rep has no GIR by design)",
    )


def plot_tr_by_split_progression(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """split_tr restricted to one dataset's tasks, x compressed to consecutive
    dataset-task number (no flat gaps), faceted by data-split category."""
    from analysis.multirep_aggregate_plots import _dataset_task_order, _add_behavior_system_legend
    order = _dataset_task_order(pair, task_type)
    cats = _present_categories(pair)
    fig, axes = plt.subplots(1, len(cats), figsize=(5.2 * len(cats), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, cat in zip(axes, cats):
        for system, exp in pair.items():
            rep = _with_category(exp.reputation_timeline())
            rep = rep[(rep["split"] == cat) & (rep["task_index"].isin(order))].copy()
            if rep.empty:
                continue
            if exp.system == "globalrep":
                rep["_v"] = rep["tr_post"]
            else:
                rep["_v"] = rep["tr_all_post"].apply(
                    lambda d: d.get(task_type) if isinstance(d, dict) else None)
            rep["_x"] = rep["task_index"].map(order)
            agg = rep.dropna(subset=["_v"]).groupby(["behavior", "_x"])["_v"].mean().reset_index()
            for behavior, grp in agg.groupby("behavior"):
                grp = grp.sort_values("_x")
                ax.plot(grp["_x"], grp["_v"], color=BEHAVIOR_COLORS.get(behavior, "#888"),
                        ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
        ax.set_title(cat)
        ax.set_xlabel(f"{TASK_TYPE_LABELS[task_type]} task #")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Task Reputation (TR)")
    _add_behavior_system_legend(axes[-1], pair)
    fig.suptitle(f"{TASK_TYPE_LABELS[task_type]}-only task-reputation by data-split "
                 "(consecutive dataset tasks; colour=behavior, style=system)")
    fig.tight_layout()
    return fig


# =========================================================================
# 4. Final-accuracy contribution by dominant selected data-split
# =========================================================================

def plot_final_accuracy_by_dominant_split(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """Group each task by the data-split category that dominated its selection,
    then show mean final accuracy per dominant category, global-rep vs multi-rep.

    Answers: 'when the protocol mostly picks CIFAR-heavy participants, does CIFAR
    accuracy actually come out higher?'
    """
    cats = _present_categories(pair)
    systems = [s for s, _ in pair.items()]
    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.6), 4.5))
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(cats))

    for i, (system, exp) in enumerate(pair.items()):
        rep = _with_category(exp.reputation_timeline())
        rep = rep[(rep["task_type"] == task_type) & (rep["was_selected"])]
        ga = exp.global_accuracy()
        if rep.empty or ga.empty:
            continue
        # dominant category per (run, task)
        dom = (rep.groupby(["run", "task_index", "split"]).size()
               .reset_index(name="n")
               .sort_values("n").groupby(["run", "task_index"]).last().reset_index())
        # final accuracy per (run, task)
        fin = (ga.sort_values("round").groupby(["run", "task_index"]).last()
               .reset_index()[["run", "task_index", "objective_global_accuracy"]])
        merged = dom.merge(fin, on=["run", "task_index"], how="inner")
        means = [merged.loc[merged["split"] == c, "objective_global_accuracy"].mean() for c in cats]
        errs = [merged.loc[merged["split"] == c, "objective_global_accuracy"].std()
                if (merged["split"] == c).sum() > 1 else 0.0 for c in cats]
        from analysis.multirep_aggregate_plots import SYSTEM_COLORS
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=errs, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Mean final accuracy of tasks")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Dominant data-split among selected participants")
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]}: final accuracy vs dominant selected data-split")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# internal
# =========================================================================

def _split_legend(ax, pair: ExperimentPair) -> None:
    behaviors = _present_behaviors(pair)
    beh_handles = [Line2D([0], [0], color=BEHAVIOR_COLORS.get(b, "#888"), lw=_LW,
                          label=BEHAVIOR_LABELS.get(b, b)) for b in behaviors]
    sys_handles = [Line2D([0], [0], color="#555", ls=SYSTEM_LS[s], lw=_LW,
                          label=SYSTEM_LABELS[s]) for s, _ in pair.items()]
    leg1 = ax.legend(handles=beh_handles, title="Behavior", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=sys_handles, title="System", loc="lower right", fontsize=8)
