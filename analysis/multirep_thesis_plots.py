"""Additional analysis graphs proposed for the thesis.

  1. Selection-score decomposition (TR / GIR / Q terms) over tasks.
  2. Task-reputation ↔ achieved-accuracy correlation.
  3. Cold-start latency: first selection for CIFAR by data-split.
  4. Selection efficiency vs a data-richness oracle.
  5. Run-to-run variability of final accuracy.
  6. Free-rider economics: net earnings per selected task + detection rate.

All compare global-rep vs multi-rep and pool the runs of each experiment.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
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
    load_partition_data_percent,
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
from analysis.multirep_grouped_plots import _present_categories, _with_category

_LW = 2
_EPS = 1e-6
_BALANCE_INITIAL = 100.0


def _weights(exp: ExperimentRuns) -> tuple[float, float, float]:
    """(tr_weight, gir_weight, q_weight) from the preset (with sane defaults)."""
    p = exp.sessions[0].preset if exp.sessions else {}
    return (float(p.get("tr_weight", 6)), float(p.get("gir_weight", 4)),
            float(p.get("q_weight", 0.0)))


# =========================================================================
# 1. Selection-score decomposition
# =========================================================================

def plot_score_decomposition(pair: ExperimentPair) -> plt.Figure:
    """Stacked TR / GIR / Q contribution to the mean selection score over tasks.

    score = (TR·tr_w + GIR·gir_w)/(tr_w+gir_w) + q_w·Q.  Shows *what* the protocol
    actually ranks on: multi-rep should lean on TR, global-rep on its single TR
    with no GIR term.
    """
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7 * len(systems), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    comp_colors = {"TR": "#1b9e77", "GIR": "#7570b3", "Q": "#e6ab02"}

    for ax, (system, exp) in zip(axes, pair.items()):
        rep = exp.reputation_timeline()
        tr_w, gir_w, q_w = _weights(exp)
        denom = tr_w + gir_w
        rep = rep.assign(
            _tr=rep["tr_pre"] * tr_w / denom,
            _gir=rep["gir_pre"] * gir_w / denom,
            _q=q_w * rep["q_pre"],
        )
        agg = rep.groupby("task_index")[["_tr", "_gir", "_q"]].mean().reset_index().sort_values("task_index")
        ax.stackplot(agg["task_index"], agg["_tr"], agg["_gir"], agg["_q"],
                     labels=["TR term", "GIR term", "Q term"],
                     colors=[comp_colors["TR"], comp_colors["GIR"], comp_colors["Q"]],
                     alpha=0.85)
        ax.set_title(f"{SYSTEM_LABELS[system]}  (tr={tr_w:g}, gir={gir_w:g}, q={q_w:g})")
        ax.set_xlabel("Task index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean selection-score contribution")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Selection-score decomposition (what drives ranking)")
    fig.tight_layout()
    return fig


# =========================================================================
# 2. Task-reputation ↔ accuracy correlation
# =========================================================================

def plot_taskrep_accuracy_correlation(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """Scatter: selected pool's mean pre-task TR vs the task's final accuracy.

    Tests whether the reputation signal the protocol selected on actually
    predicts model performance for this dataset.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for system, exp in pair.items():
        rep = exp.reputation_timeline()
        ga = exp.global_accuracy()
        if rep.empty or ga.empty:
            continue
        sel = rep[(rep["task_type"] == task_type) & (rep["was_selected"])].copy()
        if exp.system == "globalrep":
            sel["_tr"] = sel["tr_pre"]
        else:
            sel["_tr"] = sel["tr_all_pre"].apply(
                lambda d: d.get(task_type, 0.0) if isinstance(d, dict) else 0.0)
        pool = sel.groupby(["run", "task_index"])["_tr"].mean().reset_index()
        fin = (ga.sort_values("round").groupby(["run", "task_index"]).last()
               .reset_index()[["run", "task_index", "objective_global_accuracy"]])
        m = pool.merge(fin, on=["run", "task_index"], how="inner")
        if m.empty:
            continue
        ax.scatter(m["_tr"], m["objective_global_accuracy"], s=28, alpha=0.6,
                   color=SYSTEM_COLORS[system], label=SYSTEM_LABELS[system])
        if len(m) >= 3 and m["_tr"].std() > _EPS:
            r = np.corrcoef(m["_tr"], m["objective_global_accuracy"])[0, 1]
            ax.plot([], [], " ", label=f"  {SYSTEM_LABELS[system]} r={r:.2f}")
    ax.set_xlabel(f"Selected pool mean pre-task TR ({TASK_TYPE_LABELS[task_type]})")
    ax.set_ylabel("Task final accuracy")
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]}: does selected TR predict accuracy?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =========================================================================
# 3. Cold-start latency
# =========================================================================

def plot_cold_start_latency(pair: ExperimentPair, task_type: int = CIFAR_TT) -> plt.Figure:
    """Mean task index of a participant's *first* selection for `task_type`,
    grouped by data-split category.  Lower = onboarded sooner.  Tests whether
    multi-rep's per-task reputation lets data-rich newcomers in faster.
    """
    cats = _present_categories(pair)
    systems = [s for s, _ in pair.items()]
    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.5), 4.5))
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(cats))

    for i, (system, exp) in enumerate(pair.items()):
        rep = _with_category(exp.reputation_timeline())
        sub = rep[(rep["task_type"] == task_type) & (rep["was_selected"])]
        first = sub.groupby(["run", "guid", "split"])["task_index"].min().reset_index()
        means = [first.loc[first["split"] == c, "task_index"].mean() for c in cats]
        errs = [first.loc[first["split"] == c, "task_index"].std()
                if (first["split"] == c).sum() > 1 else 0.0 for c in cats]
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=errs, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel(f"First {TASK_TYPE_LABELS[task_type]} selection (task index)")
    ax.set_title(f"Cold-start latency for {TASK_TYPE_LABELS[task_type]} by data-split (lower = sooner)")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 4. Selection efficiency vs data-richness oracle
# =========================================================================

def plot_selection_efficiency(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """Overlap between the protocol's top-N by selection *score* and the top-N by
    *data_percent* for the task's dataset (an oracle that always ranks the most
    data-rich first).  1.0 = the merit ranking is perfectly data-aware.

    Ranking is by selection score, not realised selection, so the Q-value's
    rotation of the actual picks does not confound the measurement.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.array([0.0])
    plotted = False
    for i, (system, exp) in enumerate(pair.items()):
        pct = load_partition_data_percent(exp)
        rep = exp.reputation_timeline()
        if not pct or rep.empty:
            continue
        # If everyone holds the same data_percent for this dataset (e.g. the
        # avg-distribution preset), the "data-rich" oracle is undefined and the
        # metric degenerates to N/total — skip rather than mislead.
        spread = {p.get(task_type, 0.0) for p in pct.values()}
        if len(spread) <= 1:
            continue
        sub = rep[rep["task_type"] == task_type]
        effs = []
        for (_run, _ti), g in sub.groupby(["run", "task_index"]):
            n_sel = int(g["was_selected"].sum())
            if n_sel == 0:
                continue
            # oracle: most data-rich for this dataset
            oracle = set(sorted(g["user_name"],
                                key=lambda nm: pct.get(nm, {}).get(task_type, 0.0),
                                reverse=True)[:n_sel])
            # protocol's merit ranking (rotation-free)
            top_by_score = set(g.sort_values("selection_score", ascending=False)
                               ["user_name"].head(n_sel))
            effs.append(len(top_by_score & oracle) / n_sel)
        if effs:
            ax.bar(x + (i - 0.5) * width, [np.mean(effs)], width, yerr=[np.std(effs)],
                   capsize=4, color=SYSTEM_COLORS[system], edgecolor="black",
                   linewidth=0.7, alpha=0.85, label=SYSTEM_LABELS[system])
            plotted = True
    ax.axhline(1.0, color="#2e7d32", ls="--", lw=1, alpha=0.6, label="oracle")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[task_type]])
    ax.set_ylabel("Selected ∩ data-rich top-N  /  N")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]}: data-aware selection efficiency vs oracle")
    if not plotted:
        ax.text(0.5, 0.5, "No partition data_percent available",
                ha="center", va="center", transform=ax.transAxes, color="#888")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 5. Run-to-run variability
# =========================================================================

def plot_run_variability(pair: ExperimentPair) -> plt.Figure:
    """Std of per-run mean final accuracy, per dataset — how reproducible each
    system is across the runs in FinishedRuns (smaller = more stable)."""
    datasets = [("mnist", MNIST_TT), ("cifar-10", CIFAR_TT)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(datasets))
    for i, (system, exp) in enumerate(pair.items()):
        ga = exp.global_accuracy()
        stds = []
        for ds, _tt in datasets:
            sub = ga[ga["dataset"].str.lower() == ds]
            if sub.empty:
                stds.append(np.nan); continue
            fin = sub.sort_values("round").groupby(["run", "task_index"]).last().reset_index()
            per_run = fin.groupby("run")["objective_global_accuracy"].mean()
            stds.append(per_run.std() if len(per_run) > 1 else 0.0)
        ax.bar(x - 0.4 + i * width + width / 2, stds, width,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[MNIST_TT], TASK_TYPE_LABELS[CIFAR_TT]])
    ax.set_ylabel("Std of per-run mean final accuracy")
    ax.set_title("Run-to-run variability (lower = more reproducible)")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 6. Free-rider economics
# =========================================================================

def plot_freerider_economics(pair: ExperimentPair) -> plt.Figure:
    """Two panels: (left) mean net ETH earned per *selected* task by behavior;
    (right) disqualification rate by behavior.  Surfaces whether free-riders
    profit because detection misses them.
    """
    behaviors = [b for b in BEHAVIOR_ORDER
                 if any(b in exp.reputation_timeline().get("behavior", pd.Series(dtype=str)).unique()
                        for _, exp in pair.items())]
    systems = [s for s, _ in pair.items()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(behaviors))

    for i, (system, exp) in enumerate(pair.items()):
        rep = exp.reputation_timeline()
        rep = rep.assign(_delta=rep["balance_post"] - rep["balance_pre"])
        sel = rep[rep["was_selected"]]
        earn = [sel.loc[sel["behavior"] == b, "_delta"].mean() for b in behaviors]
        ax1.bar(x - 0.4 + i * width + width / 2, earn, width,
                color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
                alpha=0.85, label=SYSTEM_LABELS[system])
        # disqualification rate from per-task users tables
        disq, part = {}, {}
        for _run, _ti, _ds, _tt, u in exp.iter_task_users():
            for un, g in u.groupby("user_number"):
                role = g["role"].iloc[0] if "role" in g.columns else g["behavior"].iloc[0]
                rn = role.name.lower() if hasattr(role, "name") else str(role).lower()
                part[rn] = part.get(rn, 0) + 1
                if (g["state"] == "disqualified").any():
                    disq[rn] = disq.get(rn, 0) + 1
        rates = [disq.get(b, 0) / part.get(b, 1) for b in behaviors]
        ax2.bar(x - 0.4 + i * width + width / 2, rates, width,
                color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
                alpha=0.85, label=SYSTEM_LABELS[system])

    ax1.axhline(0, color="#555", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax1.set_ylabel("Mean net ETH per selected task")
    ax1.set_title("Earnings per selected task by behavior")
    ax1.legend(title="System", fontsize=8); ax1.grid(True, axis="y", alpha=0.3); ax1.set_axisbelow(True)

    ax2.set_xticks(x); ax2.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in behaviors])
    ax2.set_ylabel("Disqualification rate"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Detection rate by behavior")
    ax2.legend(title="System", fontsize=8); ax2.grid(True, axis="y", alpha=0.3); ax2.set_axisbelow(True)

    fig.suptitle("Free-rider economics: do undetected free-riders out-earn honest participants?")
    fig.tight_layout()
    return fig
