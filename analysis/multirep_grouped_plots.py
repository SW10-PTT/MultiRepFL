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
    SYSTEM_COLORS,
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


# --- behaviour-role buckets (task-hopper presets) -----------------------------
# Honest users share one bucket; each malicious / free-rider role is its own.
_ROLE_ORDER = [
    "Honest",
    "Malicious (both)", "Malicious (MNIST)", "Malicious (CIFAR)",
    "Freerider (both)", "Freerider (MNIST)", "Freerider (CIFAR)",
]
ROLE_COLORS = {
    "Honest":            "#2196F3",
    "Malicious (both)":  "#d62728",
    "Malicious (MNIST)": "#ff7f0e",
    "Malicious (CIFAR)": "#8c1a1a",
    "Freerider (both)":  "#9467bd",
    "Freerider (MNIST)": "#e377c2",
    "Freerider (CIFAR)": "#5b2d8c",
}


def role_bucket(name: str) -> str:
    """Map a task-hopper participant name to its behaviour-role bucket.

    Names look like ``H3 Both`` / ``M2 MNIST`` / ``F3 CIFAR-10``: the id prefix
    gives the behaviour family (H/M/F) and the suffix the adversary dataset.
    All honest users collapse into a single ``Honest`` bucket; each malicious /
    free-rider variant keeps its own bucket.
    """
    if not name:
        return "Other"
    nl = name.strip()
    fam = {"H": "Honest", "M": "Malicious", "F": "Freerider"}.get(nl[0].upper())
    if fam is None:
        return "Other"
    if fam == "Honest":
        return "Honest"
    rest = nl.split(None, 1)[1].lower() if " " in nl else ""
    if rest.startswith("mnist"):
        scope = "MNIST"
    elif rest.startswith("cifar"):
        scope = "CIFAR"
    else:
        scope = "both"
    return f"{fam} ({scope})"


def _with_category(rep: pd.DataFrame, bucket_fn=split_category) -> pd.DataFrame:
    rep = rep.copy()
    rep["split"] = rep["user_name"].map(bucket_fn)
    return rep


def _present_categories(pair: ExperimentPair, bucket_fn=split_category) -> list[str]:
    present = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep["user_name"].map(bucket_fn).unique())
    order = _ROLE_ORDER if bucket_fn is role_bucket else _CATEGORY_ORDER
    ordered = [c for c in order if c in present]
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

def plot_selection_rate_by_split(pair: ExperimentPair, task_type: int, bucket_fn=split_category) -> plt.Figure:
    """Selection rate per (bucket, behavior); one subplot per system.

    *bucket_fn* maps a user name to its bucket — ``split_category`` (data-split)
    by default, or :func:`role_bucket` for the behaviour-role buckets.
    """
    cats = _present_categories(pair, bucket_fn)
    behaviors = _present_behaviors(pair)
    systems = [s for s, _ in pair.items()]
    is_role = bucket_fn is role_bucket
    fig, axes = plt.subplots(1, len(systems), figsize=(max(7, len(cats) * 1.2) * len(systems) / 2 + 1, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = _with_category(exp.reputation_timeline(), bucket_fn)
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
        ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Selection rate")
    axes[-1].legend(title="Behavior", fontsize=8)
    grouping = "behaviour role" if is_role else "data-split"
    fig.suptitle(f"{TASK_TYPE_LABELS[task_type]} selection rate by {grouping} & behavior")
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


def plot_net_earnings_by_split(pair: ExperimentPair, bucket_fn=split_category) -> plt.Figure:
    """Mean net ETH earned per participant, grouped by (bucket, behavior).

    Earnings are summed per-task deltas (per user, per behavior), so a
    mixed-behavior user contributes their free-rider-dataset tasks to the
    free-rider bar and their honest-dataset tasks to the honest bar — their two
    parts sum to their true total with no double count.
    """
    cats = _present_categories(pair, bucket_fn)
    behaviors = _present_behaviors(pair)
    systems = [s for s, _ in pair.items()]
    is_role = bucket_fn is role_bucket
    fig, axes = plt.subplots(1, len(systems), figsize=(max(7, len(cats) * 1.2) * len(systems) / 2 + 1, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = _per_task_delta(_with_category(exp.reputation_timeline(), bucket_fn))
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
        ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Net earnings (ETH, summed task deltas)")
    axes[-1].legend(title="Behavior", fontsize=8)
    grouping = "behaviour role" if is_role else "data-split"
    fig.suptitle(f"Net earnings by {grouping} & behavior — both datasets combined "
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

def _mixed_user_table(exp, include_honest: bool = False) -> pd.DataFrame:
    """Per (user, task_type): mean net earnings across runs + the user's behavior
    on that dataset. Restricted to users whose behavior differs across datasets;
    when *include_honest* is set, all-honest users are also kept (for reference)."""
    rep = _per_task_delta(exp.reputation_timeline())
    per_run = (rep.groupby(["run", "user_name", "task_type"])
               .agg(delta=("_delta", "sum"), behavior=("behavior", "first"))
               .reset_index())
    agg = (per_run.groupby(["user_name", "task_type"])
           .agg(delta=("delta", "mean"), behavior=("behavior", "first"))
           .reset_index())
    nbeh = agg.groupby("user_name")["behavior"].nunique()
    keep = set(nbeh[nbeh > 1].index)
    if include_honest:
        all_honest = agg.groupby("user_name")["behavior"].agg(lambda s: set(s) == {"honest"})
        keep |= set(all_honest[all_honest].index)
    return agg[agg["user_name"].isin(keep)]


def plot_mixed_behavior_users(pair: ExperimentPair, include_honest: bool = False) -> plt.Figure:
    """Per-dataset net earnings for users whose behavior differs across datasets
    (e.g. free-rider on MNIST, honest on CIFAR).  Each user gets one bar per
    dataset, coloured by the behavior they hold on that dataset — showing the
    same identity treated correctly per dataset.  With *include_honest*, all-honest
    users are shown too as a baseline reference.
    """
    systems = [s for s, _ in pair.items()]
    tables = {s: _mixed_user_table(exp, include_honest) for s, exp in pair.items()}
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
    extra = " + honest baseline" if include_honest else ""
    fig.suptitle(f"Mixed-behavior users{extra}: per-dataset earnings (bar letter = M[NIST]/C[IFAR]; "
                 "colour = behavior on that dataset)")
    fig.tight_layout()
    return fig


# =========================================================================
# 2d. Task-hoppers — reputation & selection development over the session
#     (only meaningful for the task-hopper presets, where the same identity is
#      honest on one dataset and an adversary on the other)
# =========================================================================

def _taskhopper_sides(exp) -> dict:
    """Map each task-hopper to its honest/adversary datasets.

    A task-hopper is a user_name whose behaviour differs across task types and
    that is honest on exactly one dataset and an adversary on the other(s).
    Returns {user_name: {"honest_tt": int, "adv_tt": int, "adv_behavior": str}}.
    """
    rep = exp.reputation_timeline()
    if rep.empty:
        return {}
    beh = (rep.groupby(["user_name", "task_type"])["behavior"]
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
           .reset_index())
    nbeh = beh.groupby("user_name")["behavior"].nunique()
    sides = {}
    for u in nbeh[nbeh > 1].index:
        sub = beh[beh["user_name"] == u]
        honest = sub[sub["behavior"] == "honest"]["task_type"].tolist()
        adv = sub[sub["behavior"] != "honest"]["task_type"].tolist()
        if len(honest) == 1 and len(adv) >= 1:
            adv_tt = int(adv[0])
            sides[u] = {
                "honest_tt": int(honest[0]),
                "adv_tt": adv_tt,
                "adv_behavior": sub[sub["task_type"] == adv_tt]["behavior"].iat[0],
            }
    return sides


def _taskhoppers_present(pair: ExperimentPair) -> bool:
    return any(_taskhopper_sides(exp) for _, exp in pair.items())


def _side_tr(rep: pd.DataFrame, exp, sides: dict, which: str) -> pd.Series:
    """Per-row TR on the user's honest- or adversary-side dataset.

    global-rep has one shared bucket (tr_post); multi-rep reads the per-task-type
    value for the side's task type — carried on every row via tr_all_post."""
    def _one(row):
        tt = sides[row["user_name"]][which]
        if exp.system == "globalrep":
            return row["tr_post"]
        d = row["tr_all_post"]
        return d.get(tt) if isinstance(d, dict) else None
    return rep.apply(_one, axis=1)


def plot_taskhopper_reputation_development(pair: ExperimentPair) -> plt.Figure:
    """How a task-hopper's reputation evolves over the session: honest-side TR,
    adversary-side TR, and GIR.  Multi-rep should keep honest-side TR high while
    adversary-side TR decays; global-rep blends both into one polluted bucket
    (its honest- and adversary-side lines are identical by construction)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    if not _taskhoppers_present(pair):
        for ax in axes:
            ax.text(0.5, 0.5, "No task-hoppers in this experiment",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
        fig.suptitle("Task-hopper reputation development (none present)")
        fig.tight_layout()
        return fig

    panels = [("honest_tt", "Honest-side Task Reputation"),
              ("adv_tt", "Adversary-side Task Reputation")]
    ds_handles = None
    for ax, (which, title) in zip(axes[:2], panels):
        ds_handles = _mark_dataset_switches(ax, pair)
        for system, exp in pair.items():
            sides = _taskhopper_sides(exp)
            rep = exp.reputation_timeline()
            rep = rep[rep["user_name"].isin(sides)].copy()
            if rep.empty:
                continue
            rep["_v"] = _side_tr(rep, exp, sides, which)
            agg = (rep.dropna(subset=["_v"]).groupby("task_index")["_v"]
                   .agg(["mean", "std"]).reset_index().sort_values("task_index"))
            ax.plot(agg["task_index"], agg["mean"], color=SYSTEM_COLORS[system],
                    ls=SYSTEM_LS[system], lw=_LW, label=SYSTEM_LABELS[system])
            ax.fill_between(agg["task_index"], agg["mean"] - agg["std"].fillna(0),
                            agg["mean"] + agg["std"].fillna(0),
                            color=SYSTEM_COLORS[system], alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("Task index")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    # GIR panel
    ax = axes[2]
    _mark_dataset_switches(ax, pair)
    for system, exp in pair.items():
        sides = _taskhopper_sides(exp)
        rep = exp.reputation_timeline()
        rep = rep[rep["user_name"].isin(sides)]
        if rep.empty:
            continue
        agg = (rep.groupby("task_index")["gir_post"]
               .agg(["mean", "std"]).reset_index().sort_values("task_index"))
        ax.plot(agg["task_index"], agg["mean"], color=SYSTEM_COLORS[system],
                ls=SYSTEM_LS[system], lw=_LW, label=SYSTEM_LABELS[system])
        ax.fill_between(agg["task_index"], agg["mean"] - agg["std"].fillna(0),
                        agg["mean"] + agg["std"].fillna(0),
                        color=SYSTEM_COLORS[system], alpha=0.12)
    ax.set_title("Global Integrity Reputation")
    ax.set_xlabel("Task index")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Reputation")
    sys_handles = [Line2D([0], [0], color=SYSTEM_COLORS[s], ls=SYSTEM_LS[s], lw=_LW,
                          label=SYSTEM_LABELS[s]) for s, _ in pair.items()]
    leg = axes[-1].legend(handles=sys_handles, title="System", loc="upper right", fontsize=8)
    if ds_handles:
        axes[-1].add_artist(leg)
        axes[-1].legend(handles=ds_handles, title="Dataset (tint)", loc="lower right", fontsize=7)
    fig.suptitle("Task-hopper reputation development "
                 "(mean over mixed-behaviour users ±1σ; style=system)")
    fig.tight_layout()
    return fig


def plot_taskhopper_selection_development(pair: ExperimentPair) -> plt.Figure:
    """Cumulative selection rate of task-hoppers over the session, split by
    whether the running task is on their honest-side or adversary-side dataset.

    Multi-rep should keep picking them for their honest dataset while dropping
    them on their adversary dataset; global-rep cannot separate the two."""
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7 * len(systems), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    if not _taskhoppers_present(pair):
        for ax in axes:
            ax.text(0.5, 0.5, "No task-hoppers in this experiment",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
        fig.suptitle("Task-hopper selection development (none present)")
        fig.tight_layout()
        return fig

    side_colors = {"honest-side": BEHAVIOR_COLORS["honest"], "adversary-side": BEHAVIOR_COLORS["malicious"]}
    for ax, (system, exp) in zip(axes, pair.items()):
        _mark_dataset_switches(ax, pair)
        sides = _taskhopper_sides(exp)
        rep = exp.reputation_timeline()
        rep = rep[rep["user_name"].isin(sides)].copy()
        if not rep.empty:
            rep["_side"] = rep.apply(
                lambda r: "honest-side" if r["task_type"] == sides[r["user_name"]]["honest_tt"]
                else "adversary-side", axis=1)
            for side, grp in rep.groupby("_side"):
                # cumulative selection rate over task_index (selections / opportunities)
                per_task = (grp.groupby("task_index")["was_selected"]
                            .agg(["sum", "count"]).reset_index().sort_values("task_index"))
                cum_rate = per_task["sum"].cumsum() / per_task["count"].cumsum()
                ax.plot(per_task["task_index"], cum_rate, color=side_colors[side],
                        lw=_LW, marker="o", ms=3, label=side)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xlabel("Task index")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cumulative selection rate")
    handles = [Line2D([0], [0], color=c, lw=_LW, marker="o", ms=4, label=s)
               for s, c in side_colors.items()]
    axes[-1].legend(handles=handles, title="Dataset side", fontsize=8)
    fig.suptitle("Task-hopper selection development "
                 "(cumulative pick rate on their honest vs adversary dataset)")
    fig.tight_layout()
    return fig


_DS_COLORS = {MNIST_TT: "#2196F3", CIFAR_TT: "#FF9800"}


def _tr_on_tt(rep: pd.DataFrame, exp, task_type: int) -> pd.Series:
    """TR on *task_type*: per-task-type bucket for multi-rep, the single shared
    bucket (tr_post) for global-rep (so its two dataset lines coincide)."""
    if exp.system == "globalrep":
        return rep["tr_post"]
    return rep["tr_all_post"].apply(lambda d: d.get(task_type) if isinstance(d, dict) else None)


def plot_mixed_behavior_tr_development(pair: ExperimentPair) -> plt.Figure:
    """Per mixed-behaviour user (honest on one dataset, adversary on the other):
    MNIST-TR and CIFAR-TR over the session, global-rep vs multi-rep.

    Multi-rep keeps the two datasets' TR separate (high on the honest dataset, low
    on the adversary one); global-rep has a single shared bucket, so its MNIST and
    CIFAR lines coincide — it cannot tell the two roles apart.
    """
    sides: dict = {}
    for _, exp in pair.items():
        sides.update(_taskhopper_sides(exp))
    users = sorted(sides)
    if not users:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No mixed-behaviour users in this experiment",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        fig.suptitle("Mixed-behaviour TR development (none present)")
        fig.tight_layout()
        return fig

    ncol = min(4, len(users))
    nrow = int(np.ceil(len(users) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ui, u in enumerate(users):
        ax = axes[ui // ncol][ui % ncol]
        _mark_dataset_switches(ax, pair)
        for system, exp in pair.items():
            rep = exp.reputation_timeline()
            rep = rep[rep["user_name"] == u]
            if rep.empty:
                continue
            for tt in (MNIST_TT, CIFAR_TT):
                val = _tr_on_tt(rep, exp, tt)
                agg = (rep.assign(_v=val).dropna(subset=["_v"])
                       .groupby("task_index")["_v"].mean().reset_index().sort_values("task_index"))
                ax.plot(agg["task_index"], agg["_v"], color=_DS_COLORS[tt],
                        ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
        adv = sides[u]["adv_tt"]
        ax.set_title(f"{u}  (adversary on {TASK_TYPE_LABELS[adv]})", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(True, alpha=0.3)
    for j in range(len(users), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    ds_handles = [Line2D([0], [0], color=_DS_COLORS[tt], lw=_LW, label=TASK_TYPE_LABELS[tt])
                  for tt in (MNIST_TT, CIFAR_TT)]
    sys_handles = [Line2D([0], [0], color="#555", ls=SYSTEM_LS[s], lw=_LW, label=SYSTEM_LABELS[s])
                   for s, _ in pair.items()]
    leg1 = axes[0][-1].legend(handles=ds_handles, title="TR dataset", loc="upper right", fontsize=7)
    axes[0][-1].add_artist(leg1)
    axes[0][0].legend(handles=sys_handles, title="System", loc="upper left", fontsize=7)
    fig.suptitle("Mixed-behaviour users: per-dataset task-reputation development "
                 "(colour=dataset bucket, style=system)")
    fig.supxlabel("Task index"); fig.supylabel("Task Reputation (TR)")
    fig.tight_layout()
    return fig


def _adversary_roles_present(pair: ExperimentPair) -> list[str]:
    present: set = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep["user_name"].map(role_bucket).unique())
    return [r for r in _ROLE_ORDER if r in present and r != "Honest"]


def plot_taskhopper_reputation_development_by_role(pair: ExperimentPair) -> plt.Figure:
    """Extra version of the task-hopper reputation development that breaks the
    adversaries out by *type*: MNIST-TR, CIFAR-TR and GIR over the session, one
    line per malicious / free-rider role (colour), global-rep vs multi-rep (style).
    """
    roles = _adversary_roles_present(pair)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    if not roles:
        for ax in axes:
            ax.text(0.5, 0.5, "No adversary roles in this experiment",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
        fig.suptitle("Task-hopper reputation development by role (none present)")
        fig.tight_layout()
        return fig

    panels = [("MNIST Task Reputation", MNIST_TT, "tr"),
              ("CIFAR-10 Task Reputation", CIFAR_TT, "tr"),
              ("Global Integrity Reputation", None, "gir")]
    ds_handles = None
    for ax, (title, tt, kind) in zip(axes, panels):
        ds_handles = _mark_dataset_switches(ax, pair)
        for system, exp in pair.items():
            rep = exp.reputation_timeline()
            if rep.empty:
                continue
            rep = rep.copy()
            rep["role"] = rep["user_name"].map(role_bucket)
            rep = rep[rep["role"].isin(roles)]
            if kind == "gir":
                rep["_v"] = rep["gir_post"]
            else:
                rep["_v"] = _tr_on_tt(rep, exp, tt)
            agg = (rep.dropna(subset=["_v"]).groupby(["role", "task_index"])["_v"]
                   .mean().reset_index())
            for role, grp in agg.groupby("role"):
                grp = grp.sort_values("task_index")
                ax.plot(grp["task_index"], grp["_v"], color=ROLE_COLORS.get(role, "#888"),
                        ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Task index")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Reputation")
    role_handles = [Line2D([0], [0], color=ROLE_COLORS.get(r, "#888"), lw=_LW, label=r) for r in roles]
    sys_handles = [Line2D([0], [0], color="#555", ls=SYSTEM_LS[s], lw=_LW, label=SYSTEM_LABELS[s])
                   for s, _ in pair.items()]
    leg1 = axes[-1].legend(handles=role_handles, title="Adversary role", loc="upper right", fontsize=7)
    axes[-1].add_artist(leg1)
    leg2 = axes[0].legend(handles=sys_handles, title="System", loc="upper left", fontsize=8)
    if ds_handles:
        axes[1].add_artist(axes[1].legend(handles=ds_handles, title="Dataset (tint)", loc="upper left", fontsize=7))
    fig.suptitle("Task-hopper reputation development by adversary type "
                 "(colour=role, style=system)")
    fig.tight_layout()
    return fig


def plot_selections_by_role_dataset(pair: ExperimentPair) -> plt.Figure:
    """Total selections per behaviour-role bucket, split by dataset (MNIST vs
    CIFAR), one subplot per system.  Bucket labels carry the user count, since the
    Honest bucket pools many users while each adversary role is a single user."""
    cats = _present_categories(pair, role_bucket)
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(max(8, len(cats) * 1.2) * len(systems) / 2 + 1, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ds_types = [MNIST_TT, CIFAR_TT]
    n_users = {}
    for ax, (system, exp) in zip(axes, pair.items()):
        rep = exp.reputation_timeline()
        rep = rep.copy()
        rep["role"] = rep["user_name"].map(role_bucket)
        for c in cats:
            n_users[c] = rep[rep["role"] == c]["user_name"].nunique()
        x = np.arange(len(cats))
        width = 0.8 / len(ds_types)
        for j, tt in enumerate(ds_types):
            totals = [int(rep[(rep["role"] == c) & (rep["task_type"] == tt)]["was_selected"].sum())
                      for c in cats]
            ax.bar(x - 0.4 + j * width + width / 2, totals, width, color=_DS_COLORS[tt],
                   edgecolor="black", linewidth=0.6, label=TASK_TYPE_LABELS[tt])
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={n_users.get(c, 0)})" for c in cats], rotation=20, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Total selections (count, summed over runs)")
    axes[-1].legend(title="Task dataset", fontsize=8)
    fig.suptitle("Total selections by user type & dataset "
                 "(Honest pools many users — see per-bucket n)")
    fig.tight_layout()
    return fig


# =========================================================================
# 3. TR / GIR development faceted by data-split category
# =========================================================================

def _facet_development(pair: ExperimentPair, value_fn, ylabel: str, title: str,
                       task_type: int | None = None, bucket_fn=split_category) -> plt.Figure:
    cats = _present_categories(pair, bucket_fn)
    fig, axes = plt.subplots(1, len(cats), figsize=(4.4 * len(cats), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ds_handles = None
    for ax, cat in zip(axes, cats):
        ds_handles = _mark_dataset_switches(ax, pair)
        for system, exp in pair.items():
            rep = _with_category(exp.reputation_timeline(), bucket_fn)
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
    _split_legend(axes[-1], pair, ds_handles)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _tr_value(rep: pd.DataFrame, exp, task_type: int | None):
    """TR series: global-rep uses the single shared bucket (tr_post); multi-rep
    uses the per-task-type value for *task_type*."""
    if exp.system == "globalrep" or task_type is None:
        return rep["tr_post"]
    return rep["tr_all_post"].apply(lambda d: d.get(task_type) if isinstance(d, dict) else None)


def plot_tr_by_split(pair: ExperimentPair, task_type: int, bucket_fn=split_category) -> plt.Figure:
    grouping = "behaviour role" if bucket_fn is role_bucket else "data-split"
    return _facet_development(
        pair, _tr_value, "Task Reputation (TR)",
        f"{TASK_TYPE_LABELS[task_type]} task-reputation by {grouping} "
        f"(colour=behavior, style=system)",
        task_type=task_type, bucket_fn=bucket_fn,
    )


def plot_gir_by_split(pair: ExperimentPair, bucket_fn=split_category) -> plt.Figure:
    grouping = "behaviour role" if bucket_fn is role_bucket else "data-split"
    return _facet_development(
        pair, lambda rep, exp, tt: rep["gir_post"], "Global Integrity Reputation (GIR)",
        f"GIR by {grouping} (multi-rep only; global-rep has no GIR by design)",
        bucket_fn=bucket_fn,
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

def plot_final_accuracy_by_dominant_split(pair: ExperimentPair, task_type: int, bucket_fn=split_category) -> plt.Figure:
    """Group each task by the bucket that dominated its selection, then show mean
    final accuracy per dominant bucket, global-rep vs multi-rep.

    Answers: 'when the protocol mostly picks CIFAR-heavy participants, does CIFAR
    accuracy actually come out higher?'
    """
    cats = _present_categories(pair, bucket_fn)
    systems = [s for s, _ in pair.items()]
    is_role = bucket_fn is role_bucket
    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.6), 4.5))
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(cats))

    for i, (system, exp) in enumerate(pair.items()):
        rep = _with_category(exp.reputation_timeline(), bucket_fn)
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
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean final accuracy of tasks")
    ax.set_ylim(0, 1.05)
    grouping = "behaviour role" if is_role else "data-split"
    ax.set_xlabel(f"Dominant {grouping} among selected participants")
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]}: final accuracy vs dominant selected {grouping}")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# internal
# =========================================================================

def _split_legend(ax, pair: ExperimentPair, ds_handles: list | None = None) -> None:
    behaviors = _present_behaviors(pair)
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
