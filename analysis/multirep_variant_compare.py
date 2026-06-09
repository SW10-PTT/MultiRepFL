"""Compare two *same-system* multirep experiment variants (e.g. Q-slot cap on vs
off) that the globalrep/multirep auto-pairing does not pair.

Every function takes ``variants`` — a list of two dicts::

    {"exp": ExperimentRuns, "label": str, "color": str, "ls": str}

and returns a matplotlib Figure.  The first variant is drawn solid, the second
dashed (caller may override via ``ls``).  Behaviour keeps its canonical colour in
per-behaviour graphs; the variant is then the line style / bar group.

Designed for the task-hopper Q-slot-cap comparison but generic over any two
runs of the same system.
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
    ExperimentRuns,
)
from analysis.multirep_aggregate_plots import (
    _coverage_curve,
    _gini,
    _idle_streak_table,
    _mark_dataset_switches,  # noqa: F401 (kept for parity; dataset tint needs a pair)
)
from analysis.multirep_grouped_plots import _per_task_delta, _taskhopper_sides
from analysis.multirep_plots import BEHAVIOR_COLORS, BEHAVIOR_LABELS

_LW = 2
BEHAVIOR_ORDER = ["honest", "malicious", "freerider", "inactive"]
DATASET_TINT = {MNIST_TT: "#2196F3", CIFAR_TT: "#FF9800"}


# =========================================================================
# helpers
# =========================================================================

def _present_behaviors(variants: list[dict]) -> list[str]:
    present: set = set()
    for v in variants:
        rep = v["exp"].reputation_timeline()
        if not rep.empty:
            present |= set(rep["behavior"].unique())
    ordered = [b for b in BEHAVIOR_ORDER if b in present]
    ordered += [b for b in sorted(present) if b not in ordered]
    return ordered


def _user_behavior_on_tt(rep: pd.DataFrame, task_type: int) -> pd.Series:
    """Each user's (modal) behaviour on a given task type — handles task-hoppers
    whose behaviour differs across datasets."""
    sub = rep[rep["task_type"] == task_type]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.groupby("user_name")["behavior"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])


def _tr_dev_rows(exp: ExperimentRuns, task_type: int) -> pd.DataFrame:
    """Per (task_index, run, user): the user's TR on *task_type* (carried on every
    row via tr_all_post, so it is continuous across the session) plus the user's
    behaviour on that task type.  Multi-rep only (both variants are multi-rep)."""
    rep = exp.reputation_timeline()
    if rep.empty:
        return pd.DataFrame(columns=["task_index", "_beh", "_v"])
    beh = _user_behavior_on_tt(rep, task_type)
    rep = rep.copy()
    rep["_beh"] = rep["user_name"].map(beh)
    rep["_v"] = rep["tr_all_post"].apply(
        lambda d: d.get(task_type) if isinstance(d, dict) else None)
    return rep.dropna(subset=["_v", "_beh"])


def _mark_switches_from_exp(ax, exp: ExperimentRuns) -> list:
    """Tint background per dataset and mark switches, derived from one variant's
    task order (both variants share the same task schedule)."""
    rep = exp.reputation_timeline()
    if rep.empty:
        return []
    order = (rep.drop_duplicates("task_index").sort_values("task_index")
             [["task_index", "task_type"]])
    seq = list(zip(order["task_index"], order["task_type"]))
    prev = None
    for ti, tt in seq:
        ax.axvspan(ti - 0.5, ti + 0.5, color=DATASET_TINT.get(tt, "#999"),
                   alpha=0.06, zorder=0)
        if prev is not None and tt != prev:
            ax.axvline(ti - 0.5, color="#555", ls=":", lw=1, alpha=0.5, zorder=1)
        prev = tt
    return [Patch(facecolor=DATASET_TINT.get(tt, "#999"), alpha=0.25,
                  label=TASK_TYPE_LABELS.get(tt, str(tt)))
            for tt in dict.fromkeys(tt for _, tt in seq)]


def _variant_legend(ax, variants: list[dict], extra: list | None = None) -> None:
    handles = [Line2D([0], [0], color=v["color"], ls=v["ls"], lw=_LW, label=v["label"])
               for v in variants]
    leg = ax.legend(handles=handles, title="Variant", loc="best", fontsize=8)
    if extra:
        ax.add_artist(leg)
        ax.legend(handles=extra, title="Dataset (tint)", loc="lower right", fontsize=7)


# =========================================================================
# 1. Selection rate by behaviour group
# =========================================================================

def plot_selection_rate_by_behavior(variants: list[dict], task_type: int | None = None) -> plt.Figure:
    """Mean selection rate per behaviour group, bars grouped by variant.

    For task-hoppers the behaviour is read per task row, so a user's honest-side
    and adversary-side tasks land in the right bars.  Error bars are ±1σ across
    runs.  Pass *task_type* to restrict to one dataset."""
    behaviors = _present_behaviors(variants)
    fig, ax = plt.subplots(figsize=(max(7, len(behaviors) * 1.6), 4.6))
    width = 0.8 / max(1, len(variants))
    x = np.arange(len(behaviors))
    for i, v in enumerate(variants):
        rep = v["exp"].reputation_timeline()
        if task_type is not None:
            rep = rep[rep["task_type"] == task_type]
        means, errs = [], []
        for b in behaviors:
            sub = rep[rep["behavior"] == b]
            if sub.empty:
                means.append(0.0); errs.append(0.0); continue
            per_run = sub.groupby("run")["was_selected"].mean()
            means.append(per_run.mean())
            errs.append(per_run.std() if per_run.size > 1 else 0.0)
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=errs, capsize=4,
               color=v["color"], edgecolor="black", linewidth=0.7, alpha=0.9, label=v["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Selection rate")
    ax.set_ylim(0, 1.05)
    scope = "" if task_type is None else f" — {TASK_TYPE_LABELS[task_type]} tasks"
    ax.set_title(f"Selection rate by behaviour{scope} (±1σ across runs)")
    ax.legend(title="Variant")
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_selection_rate_individual(variants: list[dict]) -> plt.Figure:
    """Per-user selection rate, bars grouped by variant, users ordered and
    coloured by their dominant behaviour.  Shows which individuals the Q-slot cap
    promotes or starves."""
    # union of users + each user's dominant behaviour (from variant 0)
    rep0 = variants[0]["exp"].reputation_timeline()
    if rep0.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
    dom = rep0.groupby("user_name")["behavior"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
    users = sorted(dom.index, key=lambda u: (BEHAVIOR_ORDER.index(dom[u])
                   if dom[u] in BEHAVIOR_ORDER else 99, u))
    fig, ax = plt.subplots(figsize=(max(9, len(users) * 0.55), 4.8))
    width = 0.8 / max(1, len(variants))
    x = np.arange(len(users))
    for i, v in enumerate(variants):
        rep = v["exp"].reputation_timeline()
        rate = rep.groupby("user_name")["was_selected"].mean()
        vals = [rate.get(u, 0.0) for u in users]
        ax.bar(x - 0.4 + i * width + width / 2, vals, width,
               color=v["color"], edgecolor="black", linewidth=0.6, alpha=0.9, label=v["label"])
    # behaviour colour strip under the axis
    for xi, u in zip(x, users):
        ax.plot([xi - 0.4, xi + 0.4], [-0.04, -0.04],
                color=BEHAVIOR_COLORS.get(dom[u], "#888"), lw=4, solid_capstyle="butt",
                clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(users, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Selection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-user selection rate (under-bar strip = dominant behaviour)")
    beh_handles = [Patch(facecolor=BEHAVIOR_COLORS.get(b, "#888"), label=BEHAVIOR_LABELS.get(b, b))
                   for b in _present_behaviors(variants)]
    var_handles = [Patch(facecolor=v["color"], edgecolor="black", label=v["label"]) for v in variants]
    leg = ax.legend(handles=var_handles, title="Variant", loc="upper right", fontsize=8)
    ax.add_artist(leg)
    ax.legend(handles=beh_handles, title="Behaviour", loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 2. Task-reputation development by behaviour group
# =========================================================================

def plot_tr_development_by_behavior(variants: list[dict], task_type: int) -> plt.Figure:
    """TR on *task_type* over the session, one line per (behaviour, variant):
    colour = behaviour, style = variant.  Continuous because tr_all_post carries
    every task type's current value on every row."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ds_handles = _mark_switches_from_exp(ax, variants[0]["exp"])
    for v in variants:
        rows = _tr_dev_rows(v["exp"], task_type)
        if rows.empty:
            continue
        agg = rows.groupby(["_beh", "task_index"])["_v"].mean().reset_index()
        for behavior, grp in agg.groupby("_beh"):
            grp = grp.sort_values("task_index")
            ax.plot(grp["task_index"], grp["_v"], color=BEHAVIOR_COLORS.get(behavior, "#888"),
                    ls=v["ls"], lw=_LW, alpha=0.9)
    ax.set_xlabel("Task index")
    ax.set_ylabel(f"{TASK_TYPE_LABELS[task_type]} Task Reputation (TR)")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    beh_handles = [Line2D([0], [0], color=BEHAVIOR_COLORS.get(b, "#888"), lw=_LW,
                          label=BEHAVIOR_LABELS.get(b, b)) for b in _present_behaviors(variants)]
    var_handles = [Line2D([0], [0], color="#555", ls=v["ls"], lw=_LW, label=v["label"]) for v in variants]
    leg1 = ax.legend(handles=beh_handles, title="Behaviour", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=var_handles, title="Variant", loc="lower right", fontsize=8)
    if ds_handles:
        ax.add_artist(leg2)
        ax.legend(handles=ds_handles, title="Dataset (tint)", loc="lower center", fontsize=7)
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]} task-reputation development by behaviour "
                 "(colour=behaviour, style=variant)")
    fig.tight_layout()
    return fig


def plot_tr_development_individual(variants: list[dict], task_type: int) -> plt.Figure:
    """Small-multiples: one panel per user, TR on *task_type* over the session,
    one line per variant.  Panel title coloured by the user's behaviour on that
    dataset."""
    # gather rows (keeping user_name + behaviour-on-dataset) and the user union
    uset: set = set()
    rows_by_v = []
    for v in variants:
        rep = v["exp"].reputation_timeline()
        beh = _user_behavior_on_tt(rep, task_type)
        rep = rep.copy()
        rep["_beh"] = rep["user_name"].map(beh)
        rep["_v"] = rep["tr_all_post"].apply(lambda d: d.get(task_type) if isinstance(d, dict) else None)
        rep = rep.dropna(subset=["_v", "_beh"])
        rows_by_v.append(rep)
        uset |= set(rep["user_name"].unique())
    users = sorted(uset)
    if not users:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig
    ncol = 5
    nrow = int(np.ceil(len(users) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    behmap = {}
    for ui, u in enumerate(users):
        ax = axes[ui // ncol][ui % ncol]
        for v, rep in zip(variants, rows_by_v):
            sub = rep[rep["user_name"] == u]
            if sub.empty:
                continue
            agg = sub.groupby("task_index")["_v"].mean().reset_index().sort_values("task_index")
            ax.plot(agg["task_index"], agg["_v"], color=v["color"], ls=v["ls"], lw=1.6, label=v["label"])
            behmap[u] = sub["_beh"].mode().iat[0] if not sub["_beh"].mode().empty else sub["_beh"].iat[0]
        ax.set_title(u, fontsize=7, color=BEHAVIOR_COLORS.get(behmap.get(u), "#333"))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    for j in range(len(users), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    var_handles = [Line2D([0], [0], color=v["color"], ls=v["ls"], lw=_LW, label=v["label"]) for v in variants]
    axes[0][-1].legend(handles=var_handles, fontsize=7, loc="upper right")
    fig.suptitle(f"{TASK_TYPE_LABELS[task_type]} per-user task-reputation development "
                 "(title colour = behaviour on this dataset)")
    fig.supxlabel("Task index"); fig.supylabel("Task Reputation (TR)")
    fig.tight_layout()
    return fig


def plot_gir_development_by_behavior(variants: list[dict]) -> plt.Figure:
    """GIR over the session by behaviour group (colour=behaviour, style=variant)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ds_handles = _mark_switches_from_exp(ax, variants[0]["exp"])
    for v in variants:
        rep = v["exp"].reputation_timeline()
        if rep.empty:
            continue
        agg = rep.groupby(["behavior", "task_index"])["gir_post"].mean().reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("task_index")
            ax.plot(grp["task_index"], grp["gir_post"], color=BEHAVIOR_COLORS.get(behavior, "#888"),
                    ls=v["ls"], lw=_LW, alpha=0.9)
    ax.set_xlabel("Task index"); ax.set_ylabel("Global Integrity Reputation (GIR)")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    beh_handles = [Line2D([0], [0], color=BEHAVIOR_COLORS.get(b, "#888"), lw=_LW,
                          label=BEHAVIOR_LABELS.get(b, b)) for b in _present_behaviors(variants)]
    var_handles = [Line2D([0], [0], color="#555", ls=v["ls"], lw=_LW, label=v["label"]) for v in variants]
    leg1 = ax.legend(handles=beh_handles, title="Behaviour", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=var_handles, title="Variant", loc="lower right", fontsize=8)
    if ds_handles:
        ax.add_artist(leg2)
        ax.legend(handles=ds_handles, title="Dataset (tint)", loc="lower center", fontsize=7)
    ax.set_title("GIR development by behaviour (colour=behaviour, style=variant)")
    fig.tight_layout()
    return fig


# =========================================================================
# 3. Economics
# =========================================================================

def plot_net_earnings_by_behavior(variants: list[dict]) -> plt.Figure:
    """Mean net ETH per participant, grouped by behaviour, bars per variant.

    Earnings are summed per-task deltas keyed to each task's own behaviour, so a
    task-hopper's honest-dataset and adversary-dataset tasks contribute to the
    correct bars."""
    behaviors = _present_behaviors(variants)
    fig, ax = plt.subplots(figsize=(max(7, len(behaviors) * 1.6), 4.6))
    width = 0.8 / max(1, len(variants))
    x = np.arange(len(behaviors))
    for i, v in enumerate(variants):
        rep = _per_task_delta(v["exp"].reputation_timeline())
        if rep.empty:
            continue
        per_user = (rep.groupby(["run", "user_name", "behavior"])["_delta"].sum().reset_index())
        means, errs = [], []
        for b in behaviors:
            sub = per_user[per_user["behavior"] == b]["_delta"]
            means.append(sub.mean() if not sub.empty else 0.0)
            errs.append(sub.std() if sub.size > 1 else 0.0)
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=errs, capsize=4,
               color=v["color"], edgecolor="black", linewidth=0.7, alpha=0.9, label=v["label"])
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax.set_ylabel("Net earnings (ETH, summed task deltas)")
    ax.set_title("Net earnings by behaviour (±1σ across users)")
    ax.legend(title="Variant")
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 4. What the Q-slot cap actually does: rotation / fairness
# =========================================================================

def plot_selection_fairness(variants: list[dict]) -> plt.Figure:
    """Participation coverage and running selection Gini over time.  The Q-slot
    cap limits how many slots the Q-bonus can claim per round, so it should trade
    rotation (coverage / equality) for merit-based selection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for v in variants:
        c = _coverage_curve(v["exp"])
        if c.empty:
            continue
        ax1.plot(c["task_index"], c["coverage"], color=v["color"], ls=v["ls"], lw=_LW,
                 marker="o", ms=3, label=v["label"])
        ax2.plot(c["task_index"], c["gini"], color=v["color"], ls=v["ls"], lw=_LW,
                 marker="o", ms=3, label=v["label"])
    ax1.set_xlabel("Task index"); ax1.set_ylabel("Fraction of users selected ≥ once")
    ax1.set_ylim(0, 1.05); ax1.set_title("Participation coverage over time")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Task index"); ax2.set_ylabel("Gini of cumulative selections")
    ax2.set_ylim(0, 1.0); ax2.set_title("Selection inequality over time")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.suptitle("Q-slot cap: rotation vs concentration of selections")
    fig.tight_layout()
    return fig


def plot_idle_streak(variants: list[dict], max_streak: int = 8) -> plt.Figure:
    """P(selected) vs idle streak, and the idle-streak distribution.  Capping the
    Q-slots weakens the long-unselected bonus, so idle users recover less."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for v in variants:
        t = _idle_streak_table(v["exp"])
        if t.empty:
            continue
        t = t.copy(); t["sc"] = t["streak"].clip(upper=max_streak)
        p = t.groupby("sc")["was_selected"].mean().reset_index()
        ax1.plot(p["sc"], p["was_selected"], color=v["color"], ls=v["ls"], lw=_LW,
                 marker="o", ms=4, label=v["label"])
        d = t.groupby("sc").size(); d = (d / d.sum()).reset_index(name="frac")
        ax2.plot(d["sc"], d["frac"], color=v["color"], ls=v["ls"], lw=_LW,
                 marker="o", ms=4, label=v["label"])
    xlbl = f"Idle streak entering task (capped at {max_streak})"
    ax1.set_xlabel(xlbl); ax1.set_ylabel("P(selected)"); ax1.set_ylim(0, 1.05)
    ax1.set_title("Selection chance vs idle streak"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel(xlbl); ax2.set_ylabel("Share of task-opportunities")
    ax2.set_title("Idle-streak distribution"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.suptitle("Q-slot cap: effect on idle-user recovery")
    fig.tight_layout()
    return fig


# =========================================================================
# 5. Accuracy
# =========================================================================

def plot_final_accuracy_per_task(variants: list[dict]) -> plt.Figure:
    """Final (last-round) global accuracy per task index, line per variant.
    De-duplicates fingerprint cache-hit clones before averaging across runs."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _mark_switches_from_exp(ax, variants[0]["exp"])
    for v in variants:
        ga = v["exp"].global_accuracy()
        if ga.empty:
            continue
        key = ["run", "fingerprint"] if "fingerprint" in ga.columns else ["run", "task_index"]
        fin = (ga.sort_values("round").groupby(key + ["task_index"]).last().reset_index()
               [["task_index", "objective_global_accuracy"]])
        agg = fin.groupby("task_index")["objective_global_accuracy"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["task_index"], agg["mean"] * 100, color=v["color"], ls=v["ls"], lw=_LW,
                marker="o", ms=4, label=v["label"])
        ax.fill_between(agg["task_index"], (agg["mean"] - agg["std"].fillna(0)) * 100,
                        (agg["mean"] + agg["std"].fillna(0)) * 100, color=v["color"], alpha=0.12)
    ax.set_xlabel("Task index"); ax.set_ylabel("Final global accuracy (%)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Variant")
    ax.set_title("Final accuracy per task (±1σ across runs)")
    fig.tight_layout()
    return fig
