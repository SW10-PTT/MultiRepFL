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

import math

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
    load_partition_data_percent,
)
from analysis.multirep_aggregate_plots import (
    BEHAVIOR_COLORS,
    BEHAVIOR_LABELS,
    BEHAVIOR_ORDER,
    SYSTEM_COLORS,
    SYSTEM_LABELS,
    SYSTEM_LS,
    _kicked_records,
    _mark_dataset_switches,
)
from analysis.multirep_grouped_plots import _present_categories, _with_category

_LW = 2
_EPS = 1e-6
_BALANCE_INITIAL = 100.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks); nan if undefined."""
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value via a normal approximation (no scipy)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se <= 0:
        return float("nan")
    t = (a.mean() - b.mean()) / se
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))


def _final_tr_per_user(exp: ExperimentRuns) -> pd.DataFrame:
    """Per user: final MNIST-TR and CIFAR-TR, averaged across runs.

    global-rep keeps one shared bucket → both columns use tr_post (so users land
    on the diagonal); multi-rep reads the per-task-type values from tr_all_post.
    Columns: user_name, behavior, tr_mnist, tr_cifar.
    """
    rep = exp.reputation_timeline()
    if rep.empty:
        return pd.DataFrame(columns=["user_name", "behavior", "tr_mnist", "tr_cifar"])
    last = rep.sort_values("task_index").groupby(["run", "user_name"]).last().reset_index()
    if exp.system == "globalrep":
        last["tr_mnist"] = last["tr_post"]
        last["tr_cifar"] = last["tr_post"]
    else:
        last["tr_mnist"] = last["tr_all_post"].apply(
            lambda d: d.get(MNIST_TT, 0.0) if isinstance(d, dict) else 0.0)
        last["tr_cifar"] = last["tr_all_post"].apply(
            lambda d: d.get(CIFAR_TT, 0.0) if isinstance(d, dict) else 0.0)
    return (last.groupby(["user_name", "behavior"])[["tr_mnist", "tr_cifar"]]
            .mean().reset_index())


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
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Mean selection-score contribution")
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
            # collapse fingerprint cache clones so per-run means are not skewed
            key = ["run", "fingerprint"] if "fingerprint" in sub.columns else ["run", "task_index"]
            fin = sub.sort_values("round").groupby(key).last().reset_index()
            per_run = fin.groupby("run")["objective_global_accuracy"].mean()
            stds.append(per_run.std() if len(per_run) > 1 else 0.0)
        ax.bar(x - 0.4 + i * width + width / 2, stds, width,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[MNIST_TT], TASK_TYPE_LABELS[CIFAR_TT]])
    ax.set_ylabel("Std of per-run mean final accuracy")
    ax.set_title("Run-to-run variability")
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
    # With no adversaries (data-split-only experiments) this graph is degenerate:
    # one "honest" bar, and the "disqualification" bar is just honest false-positive
    # kicks.  Skip with a note — the task-hopper experiments carry the adversaries.
    if len(behaviors) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "Only 'honest' participants in this experiment —\nno free-rider/malicious "
                "economics to compare.\nSee the task-hopper experiments for adversarial behaviour.",
                ha="center", va="center", transform=ax.transAxes, color="#888", fontsize=10)
        ax.axis("off")
        fig.suptitle("Free-rider economics (not applicable — adversary-free experiment)")
        fig.tight_layout()
        return fig
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
    ax2.set_title("Disqualification (detection) rate by behavior")
    ax2.legend(title="System", fontsize=8); ax2.grid(True, axis="y", alpha=0.3); ax2.set_axisbelow(True)

    fig.suptitle("Free-rider economics: do undetected free-riders out-earn honest participants?")
    fig.tight_layout()
    return fig


# =========================================================================
# 7. Contribution score vs data-richness  (reward is data-aware)
# =========================================================================

def plot_contrib_vs_data_richness(pair: ExperimentPair, task_type: int) -> plt.Figure:
    """Mean contribution score on *task_type* tasks vs the participant's data
    holding (``data_percent``) for that dataset.  contrib_score drives both TR and
    reward, so a rising trend means the protocol pays for data the participant
    actually has.  Grouped by the discrete data_percent levels in the preset.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pct = {}
    for _, exp in pair.items():
        pct = load_partition_data_percent(exp) or pct
        if pct:
            break
    levels = sorted({round(p.get(task_type, 0.0), 3) for p in pct.values()}) if pct else []
    if len(levels) < 2:
        ax.text(0.5, 0.5, "No spread in data_percent for this dataset\n(cannot relate reward to data-richness)",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_title(f"{TASK_TYPE_LABELS[task_type]}: contribution vs data-richness")
        return fig

    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(levels))
    for i, (system, exp) in enumerate(pair.items()):
        rep = exp.reputation_timeline()
        sub = rep[(rep["task_type"] == task_type) & rep["was_selected"].astype(bool)].copy()
        sub = sub.dropna(subset=["contrib_score"])
        sub["_pct"] = sub["user_name"].map(lambda nm: round(pct.get(nm, {}).get(task_type, 0.0), 3))
        means, errs = [], []
        for lv in levels:
            v = sub.loc[sub["_pct"] == lv, "contrib_score"]
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std() / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=errs, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])
        rho = _spearman(sub["_pct"].to_numpy(), sub["contrib_score"].to_numpy())
        if not math.isnan(rho):
            ax.plot([], [], " ", label=f"  {SYSTEM_LABELS[system]} ρ={rho:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv:g}" for lv in levels])
    ax.set_xlabel(f"Participant {TASK_TYPE_LABELS[task_type]} data holding (data_percent)")
    ax.set_ylabel("Mean contribution score")
    ax.set_title(f"{TASK_TYPE_LABELS[task_type]}: contribution (→ reward & TR) vs data-richness")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 8. Task-reputation cross-task transfer (specialisation signature)
# =========================================================================

def plot_tr_cross_task_transfer(pair: ExperimentPair) -> plt.Figure:
    """Per user: final MNIST-TR vs final CIFAR-TR.

    Multi-rep keeps separate per-task reputations, so a participant strong in one
    dataset and weak in the other sits *off* the diagonal (low correlation =
    specialisation).  Global-rep has a single shared bucket, so both coordinates
    are equal and every user lands on the y=x line (correlation ≈ 1).  This is the
    clearest single picture of what the two-layer system buys.
    """
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="y = x (no specialisation)")
    for system, exp in pair.items():
        df = _final_tr_per_user(exp)
        if df.empty:
            continue
        ax.scatter(df["tr_mnist"], df["tr_cifar"], s=36, alpha=0.7,
                   color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.4,
                   label=SYSTEM_LABELS[system])
        r = _spearman(df["tr_mnist"].to_numpy(), df["tr_cifar"].to_numpy())
        if not math.isnan(r):
            ax.plot([], [], " ", label=f"  {SYSTEM_LABELS[system]} ρ={r:.2f}")
    ax.set_xlabel("Final MNIST task reputation")
    ax.set_ylabel("Final CIFAR-10 task reputation")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title("Cross-task TR transfer\n(off-diagonal = specialisation; multi-rep should spread, global-rep on y=x)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# 9. Specialisation heatmap (user × task-type TR)
# =========================================================================

def plot_specialization_heatmap(pair: ExperimentPair) -> plt.Figure:
    """Heatmap of each user's final per-dataset TR, one panel per system.

    Global-rep shows two identical columns (one shared bucket); multi-rep shows
    differentiated columns where users specialise.
    """
    systems = [s for s, _ in pair.items()]
    fig, axes = plt.subplots(1, len(systems), figsize=(4.6 * len(systems), 7), squeeze=False)
    axes = axes[0]
    for ax, (system, exp) in zip(axes, pair.items()):
        df = _final_tr_per_user(exp)
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(SYSTEM_LABELS[system]); ax.axis("off"); continue
        df = df.sort_values(["behavior", "user_name"])
        mat = df[["tr_mnist", "tr_cifar"]].to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, mat.max()))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["MNIST", "CIFAR-10"])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"{n}" for n in df["user_name"]], fontsize=6)
        ax.set_title(SYSTEM_LABELS[system])
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if mat[r, c] < 0.5 * mat.max() else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Final TR")
    fig.suptitle("Per-user task-reputation by dataset "
                 "(global-rep: identical columns; multi-rep: specialised)")
    fig.tight_layout()
    return fig


# =========================================================================
# 9b. Spread of final per-user task reputation
# =========================================================================

def plot_final_tr_spread(pair: ExperimentPair, kind: str = "violin") -> plt.Figure:
    """Distribution of each user's final task-reputation, one violin/box per
    (dataset, system).

    Companion to :func:`plot_tr_cross_task_transfer` — shows how spread out
    users' final TRs are rather than the per-user pairing.  Global-rep's
    single shared bucket means its MNIST and CIFAR distributions are
    identical; multi-rep's per-task TRs may differ between datasets.
    """
    datasets = [("mnist", "tr_mnist"), ("cifar-10", "tr_cifar")]
    systems = [s for s, _ in pair.items()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(datasets))
    rng = np.random.default_rng(0)

    for i, (system, exp) in enumerate(pair.items()):
        df = _final_tr_per_user(exp)
        data = [df[col].dropna().to_numpy() if not df.empty else np.array([]) for _, col in datasets]
        positions = x - 0.4 + i * width + width / 2
        color = SYSTEM_COLORS[system]

        if kind == "violin":
            present = [(p, d) for p, d in zip(positions, data) if len(d) >= 2]
            if present:
                vp = ax.violinplot([d for _, d in present], positions=[p for p, _ in present],
                                    widths=width * 0.9, showmedians=True, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.5)
                    body.set_edgecolor("black")
                vp["cmedians"].set_color("black")
        else:
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for med in bp["medians"]:
                med.set_color("black")

        # jittered per-user points
        for p, d in zip(positions, data):
            if len(d) == 0:
                continue
            jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(d))
            ax.scatter(p + jitter, d, color="black", alpha=0.5, s=10, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[MNIST_TT], TASK_TYPE_LABELS[CIFAR_TT]])
    ax.set_ylabel("Final Task Reputation (TR)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Spread of final per-user task reputation, by dataset")
    handles = [Patch(facecolor=SYSTEM_COLORS[s], alpha=0.6, edgecolor="black", label=SYSTEM_LABELS[s])
               for s in systems]
    ax.legend(handles=handles, title="System")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 10. Final accuracy with 95% CI + significance
# =========================================================================

def plot_final_accuracy_ci(pair: ExperimentPair) -> plt.Figure:
    """Mean final accuracy per dataset with 95% CI whiskers and a Welch p-value
    between systems.  Variance is over *distinct* experimental curves (fingerprint
    cache clones collapsed), so the CI is not artificially narrow.
    """
    from analysis.multirep_aggregate_plots import _final_accuracy
    datasets = ["mnist", "cifar-10"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(datasets))
    by_ds_sys: dict = {ds: {} for ds in datasets}

    for i, (system, exp) in enumerate(pair.items()):
        fa = _final_accuracy(exp)
        means, cis = [], []
        for ds in datasets:
            vals = fa.loc[fa["dataset"] == ds, "accuracy"].to_numpy()
            by_ds_sys[ds][system] = vals
            if len(vals):
                means.append(vals.mean())
                cis.append(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan); cis.append(0.0)
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=cis, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])

    # significance annotation per dataset (global-rep vs multi-rep)
    for xi, ds in zip(x, datasets):
        sysvals = by_ds_sys[ds]
        if len(sysvals) == 2:
            (a, b) = list(sysvals.values())
            p = _welch_p(a, b)
            if not math.isnan(p):
                top = max([v.mean() for v in sysvals.values() if len(v)] + [0])
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                ax.text(xi, min(1.03, top + 0.06), f"{star}\np={p:.3f}",
                        ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[MNIST_TT], TASK_TYPE_LABELS[CIFAR_TT]])
    ax.set_ylabel("Final-round global accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Final accuracy by dataset (mean ±95% CI over distinct curves; Welch p)")
    ax.legend(title="System")
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 11. Selection-merit data-awareness (Spearman selection_score vs data_percent)
# =========================================================================

def plot_selection_merit_spearman(pair: ExperimentPair) -> plt.Figure:
    """Per task-type: mean Spearman(selection_score, data_percent) across tasks.

    +1 = the protocol's merit ranking perfectly tracks who actually holds the
    data for this dataset; 0 = blind to data-richness.  Skips a dataset with no
    data_percent spread (the oracle is then undefined).
    """
    task_types = [MNIST_TT, CIFAR_TT]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    systems = [s for s, _ in pair.items()]
    width = 0.8 / max(1, len(systems))
    x = np.arange(len(task_types))
    any_plotted = False

    for i, (system, exp) in enumerate(pair.items()):
        pct = load_partition_data_percent(exp)
        rep = exp.reputation_timeline()
        means, errs = [], []
        for tt in task_types:
            spread = {round(p.get(tt, 0.0), 3) for p in pct.values()} if pct else set()
            if not pct or len(spread) < 2 or rep.empty:
                means.append(np.nan); errs.append(0.0); continue
            sub = rep[rep["task_type"] == tt]
            rhos = []
            for (_run, _ti), g in sub.groupby(["run", "task_index"]):
                dp = g["user_name"].map(lambda nm: pct.get(nm, {}).get(tt, 0.0)).to_numpy()
                rho = _spearman(dp, g["selection_score"].to_numpy())
                if not math.isnan(rho):
                    rhos.append(rho)
            means.append(np.mean(rhos) if rhos else np.nan)
            errs.append(np.std(rhos) if len(rhos) > 1 else 0.0)
            any_plotted = any_plotted or bool(rhos)
        ax.bar(x - 0.4 + i * width + width / 2, means, width, yerr=errs, capsize=4,
               color=SYSTEM_COLORS[system], edgecolor="black", linewidth=0.7,
               alpha=0.85, label=SYSTEM_LABELS[system])
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABELS[t] for t in task_types])
    ax.set_ylabel("Spearman(selection score, data holding)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Is the merit ranking data-aware? (1 = picks track who holds the data)")
    if not any_plotted:
        ax.text(0.5, 0.5, "No data_percent spread in this experiment",
                ha="center", va="center", transform=ax.transAxes, color="#888")
    ax.legend(title="System", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# =========================================================================
# 12. Cumulative earnings by behavior (economic security)
# =========================================================================

def plot_cumulative_earnings_by_behavior(pair: ExperimentPair) -> plt.Figure:
    """Mean balance (cumulative ETH) per behavior over task index, both systems.

    The key adversarial-economics question: do free-riders / malicious users
    accumulate ETH over the session, or does the protocol bleed them dry relative
    to honest participants?  Colour = behavior, line style = system.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ds_handles = _mark_dataset_switches(ax, pair)
    for system, exp in pair.items():
        rep = exp.reputation_timeline()
        if rep.empty:
            continue
        rep = rep.assign(_bal=rep["balance_post"] + _BALANCE_INITIAL)
        agg = rep.groupby(["behavior", "task_index"])["_bal"].mean().reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("task_index")
            ax.plot(grp["task_index"], grp["_bal"],
                    color=BEHAVIOR_COLORS.get(behavior, "#888"),
                    ls=SYSTEM_LS[system], lw=_LW, alpha=0.9)
    ax.axhline(_BALANCE_INITIAL, color="#999", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Mean balance (ETH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_title("Cumulative earnings by behavior (colour=behavior, style=system; "
                 "dotted = 100 ETH break-even)")
    _earnings_legend(ax, pair, ds_handles)
    fig.tight_layout()
    return fig


# =========================================================================
# 13. Detection rate over time (adversarial)
# =========================================================================

def plot_detection_rate_over_time(pair: ExperimentPair) -> plt.Figure:
    """Fraction of adversarial participations disqualified, per task index.

    For each task, among the malicious / free-rider participants present, what
    share got kicked?  Shows whether the protocol detects adversaries reliably and
    whether detection holds as the session progresses.  Colour = behavior,
    line style = system.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ds_handles = _mark_dataset_switches(ax, pair)
    adv = ["malicious", "freerider"]
    plotted = False
    for system, exp in pair.items():
        kicks, parts = _kicked_records(exp)
        if parts.empty:
            continue
        for b in adv:
            p = parts[parts["behavior"] == b].groupby("task_index").size()
            if p.empty:
                continue
            k = (kicks[kicks["behavior"] == b].groupby("task_index").size()
                 if not kicks.empty else pd.Series(dtype=int))
            rate = (k.reindex(p.index).fillna(0) / p).sort_index()
            ax.plot(rate.index, rate.values, color=BEHAVIOR_COLORS.get(b, "#888"),
                    ls=SYSTEM_LS[system], lw=_LW, alpha=0.9, marker="o", ms=3)
            plotted = True
    ax.set_xlabel("Task index")
    ax.set_ylabel("Disqualification rate among present adversaries")
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_title("Adversary detection rate over tasks (colour=behavior, style=system)")
    if not plotted:
        ax.text(0.5, 0.5, "No adversarial participants in this experiment",
                ha="center", va="center", transform=ax.transAxes, color="#888")
    else:
        _earnings_legend(ax, pair, ds_handles, behaviors=adv)
    fig.tight_layout()
    return fig


def _earnings_legend(ax, pair: ExperimentPair, ds_handles: list | None = None,
                     behaviors: list | None = None) -> None:
    present = set()
    for _, exp in pair.items():
        rep = exp.reputation_timeline()
        if not rep.empty:
            present |= set(rep["behavior"].unique())
    behs = [b for b in (behaviors or BEHAVIOR_ORDER) if b in present]
    beh_handles = [Line2D([0], [0], color=BEHAVIOR_COLORS.get(b, "#888"), lw=_LW,
                          label=BEHAVIOR_LABELS.get(b, b)) for b in behs]
    sys_handles = [Line2D([0], [0], color="#555", ls=SYSTEM_LS[s], lw=_LW,
                          label=SYSTEM_LABELS[s]) for s, _ in pair.items()]
    leg1 = ax.legend(handles=beh_handles, title="Behavior", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=sys_handles, title="System", loc="lower right", fontsize=8)
    if ds_handles:
        ax.add_artist(leg2)
        ax.legend(handles=ds_handles, title="Dataset (tint)", loc="lower left", fontsize=7)
