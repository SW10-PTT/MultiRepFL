"""Plot functions for multirep session data.

All functions accept a ``reputation_timeline`` DataFrame (as produced by
MultirepLogger) and/or ``tasks`` list.  They return matplotlib Figure objects
so the caller can save them with ``plots.save_figure``.
"""

from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"figure.dpi": 200})

# Reuse the same role colours as the single-task plots
BEHAVIOR_COLORS = {
    "honest":    "#2196F3",
    "malicious": "#d62728",
    "freerider": "#9467bd",
}
BEHAVIOR_LABELS = {
    "honest":    "Honest",
    "malicious": "Malicious",
    "freerider": "Freerider",
}

_DEFAULT_ALPHA_FILL = 0.15
_LW = 2

# Starting ETH balance before any task ran (collateral deployed at registration).
_BALANCE_INITIAL = 100.0


def _user_color(name: str, behavior: str) -> str:
    base = BEHAVIOR_COLORS.get(behavior, "#888888")
    return base


# Colour for a participant's *data-split* leaning, used to tint name labels so
# users are distinguishable even when many lines overlap.
_SPLIT_LABEL_COLORS = {"mnist": "#1565C0", "cifar": "#E65100", "balanced": "#555555"}


def _split_label_color(name: str) -> str:
    nl = (name or "").lower()
    mnist = ("mnist-heavy" in nl or "mnist-strong" in nl
             or nl.split(" ", 1)[-1].startswith("mnist"))
    cifar = ("cifar-heavy" in nl or "cifar-strong" in nl
             or nl.split(" ", 1)[-1].startswith("cifar"))
    if mnist and not cifar:
        return _SPLIT_LABEL_COLORS["mnist"]
    if cifar and not mnist:
        return _SPLIT_LABEL_COLORS["cifar"]
    return _SPLIT_LABEL_COLORS["balanced"]


_DATASET_TINT = {5: "#2196F3", 6: "#FF9800"}  # MNIST, CIFAR-10


_DATASET_TINT_LABELS = {5: "MNIST", 6: "CIFAR-10"}


def _shade_datasets(ax, rep: pd.DataFrame) -> list:
    """Tint the background per dataset run + dashed lines at switches, so flat
    stretches (e.g. CIFAR TR during a run of MNIST tasks) are legible.

    Returns dataset-tint legend handles for the caller to display.
    """
    if "task_type" not in rep.columns:
        return []
    mp = (rep[["task_index", "task_type"]].dropna().drop_duplicates()
          .sort_values("task_index"))
    items = list(mp.itertuples(index=False))
    if not items:
        return []
    start = prev = items[0].task_index
    cur = items[0].task_type
    segs = []
    for row in items[1:]:
        if row.task_type != cur:
            segs.append((start, prev, cur))
            start, cur = row.task_index, row.task_type
        prev = row.task_index
    segs.append((start, prev, cur))
    for i, (s, e, tt) in enumerate(segs):
        ax.axvspan(s - 0.5, e + 0.5, color=_DATASET_TINT.get(int(tt), "#999"),
                   alpha=0.06, zorder=0)
        if i > 0:
            ax.axvline(s - 0.5, color="#555", ls=":", lw=1, alpha=0.5, zorder=1)
    return [matplotlib.patches.Patch(facecolor=_DATASET_TINT.get(int(tt), "#999"), alpha=0.25,
                                     label=_DATASET_TINT_LABELS.get(int(tt), str(tt)))
            for tt in dict.fromkeys(tt for _, _, tt in segs)]


def _annotate_end(ax, grp: pd.DataFrame, col: str, name: str) -> None:
    """Queue an end-of-line label for *name*; positions are de-collided later by
    :func:`_flush_end_labels` (call it once after all lines are drawn)."""
    grp = grp.dropna(subset=[col])
    if grp.empty:
        return
    last = grp.sort_values("task_index").iloc[-1]
    short = name if len(name) <= 16 else name[:15] + "…"
    if not hasattr(ax, "_end_labels"):
        ax._end_labels = []
    ax._end_labels.append((float(last["task_index"]), float(last[col]),
                           short, _split_label_color(name)))


def _flush_end_labels(ax) -> None:
    """Render queued end-of-line labels, spreading them vertically so they don't
    overlap.  Labels sit just right of the axis with a faint leader line."""
    labels = getattr(ax, "_end_labels", None)
    if not labels:
        return
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * 0.038
    labels = sorted(labels, key=lambda e: e[1])
    ys = [e[1] for e in labels]
    for i in range(1, len(ys)):
        if ys[i] < ys[i - 1] + gap:
            ys[i] = ys[i - 1] + gap
    overflow = ys[-1] - ymax
    if overflow > 0:
        ys = [y - overflow for y in ys]
    x0, x1 = ax.get_xlim()
    xl = x1 + (x1 - x0) * 0.012
    for (xpt, yorig, name, color), ynew in zip(labels, ys):
        ax.plot([xpt, xl], [yorig, ynew], color=color, lw=0.4, alpha=0.35, clip_on=False)
        ax.annotate(name, (xl, ynew), xytext=(2, 0), textcoords="offset points",
                    fontsize=6, va="center", ha="left", color=color, clip_on=False)
    ax.figure.subplots_adjust(right=0.84)
    ax._end_labels = []


def _split_label_legend(ax) -> None:
    """Add a small legend explaining the name-label colour scheme."""
    handles = [
        Line2D([0], [0], color=_SPLIT_LABEL_COLORS["mnist"], lw=0, marker="$A$", label="MNIST-leaning name"),
        Line2D([0], [0], color=_SPLIT_LABEL_COLORS["cifar"], lw=0, marker="$A$", label="CIFAR-leaning name"),
        Line2D([0], [0], color=_SPLIT_LABEL_COLORS["balanced"], lw=0, marker="$A$", label="balanced name"),
    ]
    ax.add_artist(ax.legend(handles=handles, title="Name colour", loc="lower right", fontsize=6))


def _drop_inactive(rep: pd.DataFrame) -> pd.DataFrame:
    return rep[rep["behavior"] != "inactive"]


def _with_round_zero(rep: pd.DataFrame, col_defaults: dict) -> pd.DataFrame:
    """Prepend a synthetic task=0 initial-state row per user; shift real data to start at 1.

    After this call the x-axis meaning is:
      0 = before any task ran (initial state, governed by col_defaults)
      1 = state after task 0 ran
      2 = state after task 1 ran, …
    """
    shifted = rep.copy()
    shifted["task_index"] = shifted["task_index"] + 1

    initial_rows = []
    for (user_name, behavior), grp in rep.groupby(["user_name", "behavior"], sort=False):
        row = grp.iloc[0].copy()
        row["task_index"] = 0
        row["was_selected"] = False
        for col, val in col_defaults.items():
            if col in row.index:
                row[col] = val
        initial_rows.append(row)

    return pd.concat(
        [pd.DataFrame(initial_rows), shifted], ignore_index=True
    ).sort_values(["user_name", "task_index"])


# ---------------------------------------------------------------------------
# Per-user reputation evolution over tasks
# ---------------------------------------------------------------------------

def plot_tr_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Task Reputation (TR) for every user over task index."""
    rep = _drop_inactive(rep)
    rep = _with_round_zero(rep, {"tr_post": 0.0})
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["tr_post"], color=color, linewidth=_LW, alpha=0.7,
                label=f"{name} ({BEHAVIOR_LABELS.get(behavior, behavior)})")
        _annotate_end(ax, grp, "tr_post", name)

    ax.set_xlabel("Task")
    ax.set_ylabel("Task Reputation (TR)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _flush_end_labels(ax)
    return fig


def plot_gir_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Global Integrity Reputation (GIR) for every user over task index."""
    rep = _drop_inactive(rep)
    rep = _with_round_zero(rep, {"gir_post": 0.0})
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["gir_post"], color=color, linewidth=_LW, alpha=0.7)
        _annotate_end(ax, grp, "gir_post", name)

    ax.set_xlabel("Task")
    ax.set_ylabel("Global Integrity Reputation (GIR)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _flush_end_labels(ax)
    return fig


def plot_q_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Q-value (selection pressure) for every user over task index."""
    rep = _drop_inactive(rep)
    rep = _with_round_zero(rep, {"q_post": 0.0})
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["q_post"], color=color, linewidth=_LW, alpha=0.7)

    ax.set_xlabel("Task")
    ax.set_ylabel("Q-value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_balance_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """ETH balance for every user over task index.

    user.balance stores the cumulative net delta from tasks (not including the
    100 ETH baseline).  Shift all rows by _BALANCE_INITIAL so the y-axis reads
    as absolute ETH balance, then prepend a synthetic row at 100 ETH for t=0.
    """
    rep = _drop_inactive(rep)
    rep = rep.copy()
    rep["balance_post"] = rep["balance_post"] + _BALANCE_INITIAL
    rep = _with_round_zero(rep, {"balance_post": _BALANCE_INITIAL})
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["balance_post"], color=color, linewidth=_LW, alpha=0.7)
        _annotate_end(ax, grp, "balance_post", name)

    ax.set_xlabel("Task")
    ax.set_ylabel("Balance (ETH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _flush_end_labels(ax)
    return fig


def plot_confidence_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Confidence score for every user over task index.

    Tasks with no confidence data (non-selected users, cached tasks) are treated
    as confidence=0 so the full timeline is visible.
    """
    rep = _drop_inactive(rep)
    rep = _with_round_zero(rep, {"confidence": 0.0})
    rep = rep.copy()
    rep["confidence"] = rep["confidence"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["confidence"], color=color, linewidth=_LW, alpha=0.7)
        _annotate_end(ax, grp, "confidence", name)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Confidence")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 1.05)
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _flush_end_labels(ax)
    return fig


# ---------------------------------------------------------------------------
# Group-level (mean ± std) evolution
# ---------------------------------------------------------------------------

def plot_tr_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std TR per behavior group over task index. Absolute TR for the task's task-type."""
    return _plot_metric_by_behavior(rep, "tr_post", "Task Reputation (TR)", y_min_fill=0.0)


def plot_gir_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std GIR per behavior group over task index."""
    return _plot_metric_by_behavior(rep, "gir_post", "Global Integrity Reputation (GIR)", y_min_fill=0.0)


def plot_q_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std Q-value per behavior group over task index."""
    return _plot_metric_by_behavior(rep, "q_post", "Q-value")


def plot_balance_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std balance per behavior group over task index.

    Shifts all rows by _BALANCE_INITIAL (100 ETH) so the y-axis shows absolute
    ETH balance; the synthetic t=0 row is anchored at 100 ETH.
    """
    rep = rep.copy()
    rep["balance_post"] = rep["balance_post"] + _BALANCE_INITIAL
    return _plot_metric_by_behavior(rep, "balance_post", "Balance (ETH)", initial_val=_BALANCE_INITIAL)


def _plot_metric_by_behavior(rep: pd.DataFrame, col: str, ylabel: str, initial_val: float = 0.0, add_zero_row: bool = True, y_min_fill: float | None = None) -> plt.Figure:
    rep = _drop_inactive(rep)
    if add_zero_row:
        rep = _with_round_zero(rep, {col: initial_val})
    fig, ax = plt.subplots(figsize=(9, 4))
    ds_handles = _shade_datasets(ax, rep)
    agg = (
        rep.groupby(["behavior", "task_index"])[col]
        .agg(["mean", "std"])
        .reset_index()
    )
    for behavior, grp in agg.groupby("behavior"):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        ax.plot(grp["task_index"], grp["mean"], color=color, linewidth=_LW, label=label)
        lower = grp["mean"] - grp["std"].fillna(0)
        if y_min_fill is not None:
            lower = lower.clip(lower=y_min_fill)
        ax.fill_between(
            grp["task_index"],
            lower,
            grp["mean"] + grp["std"].fillna(0),
            alpha=_DEFAULT_ALPHA_FILL, color=color,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    leg = ax.legend(title="Behavior", loc="upper left")
    if ds_handles:
        ax.add_artist(leg)
        ax.legend(handles=ds_handles, title="Dataset (tint)", loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Selected-only progression: x = n-th time the user was actually selected
# ---------------------------------------------------------------------------

def _selected_reindex(rep: pd.DataFrame, col: str) -> pd.DataFrame:
    """Re-index each user's selected tasks to 1,2,3… (their n-th selection) and
    average *col* across runs at each ordinal.

    Expects the raw run-tagged timeline (a ``run`` column).  Only tasks the user
    was actually selected for advance the x-axis, so flat unselected plateaus are
    removed.  Returns columns: user_name, behavior, nth, value.
    """
    if rep.empty or "was_selected" not in rep.columns:
        return pd.DataFrame(columns=["user_name", "behavior", "nth", col])
    sel = rep[rep["was_selected"].astype(bool)].copy()
    if sel.empty:
        return pd.DataFrame(columns=["user_name", "behavior", "nth", col])
    grp_keys = ["run", "user_name"] if "run" in sel.columns else ["user_name"]
    sel = sel.sort_values(grp_keys + ["task_index"])
    sel["nth"] = sel.groupby(grp_keys).cumcount() + 1
    return (sel.groupby(["user_name", "behavior", "nth"])[col]
            .mean().reset_index())


def _plot_selected_progression(rep: pd.DataFrame, col: str, ylabel: str,
                               y01: bool, shift: float = 0.0) -> plt.Figure:
    rep = _drop_inactive(rep)
    agg = _selected_reindex(rep, col)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if agg.empty:
        ax.text(0.5, 0.5, "No selection events", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return fig
    for (name, behavior), g in agg.groupby(["user_name", "behavior"]):
        g = g.sort_values("nth")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        yvals = g[col] + shift
        ax.plot(g["nth"], yvals, color=color, lw=_LW, alpha=0.7, marker="o", ms=3)
        end = pd.DataFrame({"task_index": g["nth"], col: yvals})
        _annotate_end(ax, end, col, name)
    ax.set_xlabel("n-th selection (unselected tasks removed)")
    ax.set_ylabel(ylabel)
    if y01:
        ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _flush_end_labels(ax)
    return fig


def plot_tr_selected_progression(rep: pd.DataFrame) -> plt.Figure:
    """Per-user TR plotted only at the tasks each user was selected for."""
    return _plot_selected_progression(rep, "tr_post", "Task Reputation (TR)", y01=True)


def plot_gir_selected_progression(rep: pd.DataFrame) -> plt.Figure:
    """Per-user GIR plotted only at the tasks each user was selected for."""
    return _plot_selected_progression(rep, "gir_post", "Global Integrity Reputation (GIR)", y01=True)


def plot_balance_selected_progression(rep: pd.DataFrame) -> plt.Figure:
    """Per-user balance plotted only at the tasks each user was selected for."""
    return _plot_selected_progression(rep, "balance_post", "Balance (ETH)",
                                      y01=False, shift=_BALANCE_INITIAL)


# ---------------------------------------------------------------------------
# Selection analysis
# ---------------------------------------------------------------------------

def plot_selection_frequency(rep: pd.DataFrame) -> plt.Figure:
    """Bar chart: how many tasks each user was selected for."""
    rep = _drop_inactive(rep)
    counts = (
        rep[rep["was_selected"]]
        .groupby(["user_name", "behavior"])
        .size()
        .reset_index(name="times_selected")
        .sort_values("times_selected", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 0.6), 4))
    colors = [BEHAVIOR_COLORS.get(b, "#888") for b in counts["behavior"]]
    ax.bar(counts["user_name"], counts["times_selected"], color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xlabel("User")
    ax.set_ylabel("Times selected")
    ax.xaxis.set_tick_params(rotation=40)
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_selection_heatmap(rep: pd.DataFrame) -> plt.Figure:
    """Heatmap: user × task_index, shaded by whether the user was selected."""
    rep = _drop_inactive(rep)
    pivot = rep.pivot_table(index="user_name", columns="task_index", values="was_selected", aggfunc="max")
    # sort rows by behavior then name for visual grouping
    order = (
        rep[["user_name", "behavior"]]
        .drop_duplicates()
        .sort_values(["behavior", "user_name"])["user_name"]
        .tolist()
    )
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.4), max(4, pivot.shape[0] * 0.35)))
    cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#2196F3"])
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Task index")
    ax.set_ylabel("User")
    ax.set_title("Selection per task (blue = selected)")

    # Dataset labels on x-axis if available (caller may add them)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Score analysis
# ---------------------------------------------------------------------------

def plot_selection_score_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Selection score for every user over tasks, marking selected tasks."""
    rep = _drop_inactive(rep)
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["selection_score"], color=color, linewidth=_LW, alpha=0.5)
        sel = grp[grp["was_selected"]]
        ax.scatter(sel["task_index"], sel["selection_score"], color=color, s=25, zorder=5)
        _annotate_end(ax, grp, "selection_score", name)

    ax.set_xlabel("Task")
    ax.set_ylabel("Selection score")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _flush_end_labels(ax)
    return fig


def plot_score_vs_tr(rep: pd.DataFrame) -> plt.Figure:
    """Scatter: selection_score vs tr_pre, coloured by behavior."""
    rep = _drop_inactive(rep)
    fig, ax = plt.subplots(figsize=(7, 5))
    for behavior, grp in rep.groupby("behavior"):
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        label = BEHAVIOR_LABELS.get(behavior, behavior)
        ax.scatter(grp["tr_pre"], grp["selection_score"], color=color, alpha=0.4, s=15, label=label)

    ax.set_xlabel("TR (pre-task)")
    ax.set_ylabel("Selection score")
    ax.legend(title="Behavior")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TR split by task type (requires tr_all_post column)
# ---------------------------------------------------------------------------

# Human-readable names for TaskType int values (mirrors TrainingSpecsJobListing.TaskType)
TASK_TYPE_LABELS = {5: "MNIST", 6: "CIFAR-10", 7: "FashionMNIST"}

# Only these task types are shown in per-task-type TR plots.
_TASK_TYPES_OF_INTEREST = {5, 6}  # MNIST=5, CIFAR-10=6


def plot_tr_per_task_type(rep: pd.DataFrame) -> plt.Figure:
    """One subplot per known task type showing each user's TR over tasks.

    Requires the ``tr_all_post`` column (dict keyed by task-type int).
    Falls back to a message if the column is absent (old session format).
    """
    rep = _drop_inactive(rep)
    if "tr_all_post" not in rep.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "tr_all_post not available — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    # Collect task types present in the session, restricted to the ones we care about.
    all_tts = sorted({
        tt
        for d in rep["tr_all_post"].dropna()
        if isinstance(d, dict)
        for tt in d
        if tt in _TASK_TYPES_OF_INTEREST
    })
    if not all_tts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No TR data in tr_all_post", ha="center", va="center", transform=ax.transAxes)
        return fig

    rep = _with_round_zero(rep, {"tr_all_post": {tt: 0.0 for tt in all_tts}})

    ncols = len(all_tts)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, tt in zip(axes, all_tts):
        label = TASK_TYPE_LABELS.get(tt, f"TaskType {tt}")
        for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
            grp = grp.sort_values("task_index")
            tr_vals = grp["tr_all_post"].apply(
                lambda d: d.get(tt) if isinstance(d, dict) else None
            )
            mask = tr_vals.notna()
            if not mask.any():
                continue
            color = BEHAVIOR_COLORS.get(behavior, "#888")
            ax.plot(grp["task_index"][mask], tr_vals[mask],
                    color=color, linewidth=_LW, alpha=0.7)

        ax.set_title(label)
        ax.set_xlabel("Task")
        ax.set_ylabel("Task Reputation (TR)")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    _add_behavior_legend(axes[-1])
    fig.tight_layout()
    return fig


def plot_tr_per_task_type_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std TR per behavior group, one subplot per task type."""
    rep = _drop_inactive(rep)
    if "tr_all_post" not in rep.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "tr_all_post not available — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    all_tts = sorted({
        tt
        for d in rep["tr_all_post"].dropna()
        if isinstance(d, dict)
        for tt in d
        if tt in _TASK_TYPES_OF_INTEREST
    })
    if not all_tts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    rep = _with_round_zero(rep, {"tr_all_post": {tt: 0.0 for tt in all_tts}})

    ncols = len(all_tts)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, tt in zip(axes, all_tts):
        label = TASK_TYPE_LABELS.get(tt, f"TaskType {tt}")
        # Expand tr_all_post into a flat column for this task type
        col = rep["tr_all_post"].apply(lambda d: d.get(tt) if isinstance(d, dict) else None)
        sub = rep.assign(_tr=col).dropna(subset=["_tr"])
        agg = sub.groupby(["behavior", "task_index"])["_tr"].agg(["mean", "std"]).reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("task_index")
            color = BEHAVIOR_COLORS.get(behavior, "#888")
            label_b = BEHAVIOR_LABELS.get(behavior, behavior)
            ax.plot(grp["task_index"], grp["mean"], color=color, linewidth=_LW, label=label_b)
            ax.fill_between(grp["task_index"],
                            grp["mean"] - grp["std"].fillna(0),
                            grp["mean"] + grp["std"].fillna(0),
                            alpha=_DEFAULT_ALPHA_FILL, color=color)
        ax.set_title(TASK_TYPE_LABELS.get(tt, f"TaskType {tt}"))
        ax.set_xlabel("Task")
        ax.set_ylabel("Task Reputation (TR)")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    axes[-1].legend(title="Behavior")
    fig.tight_layout()
    return fig


def plot_q_per_task_type(rep: pd.DataFrame) -> plt.Figure:
    """One subplot per known task type showing each user's Q-value over tasks."""
    rep = _drop_inactive(rep)
    if "q_all_post" not in rep.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "q_all_post not available — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    all_tts = sorted({
        tt
        for d in rep["q_all_post"].dropna()
        if isinstance(d, dict)
        for tt in d
        if tt in _TASK_TYPES_OF_INTEREST
    })
    if not all_tts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Q data in q_all_post", ha="center", va="center", transform=ax.transAxes)
        return fig

    rep = _with_round_zero(rep, {"q_all_post": {tt: 0.0 for tt in all_tts}})

    ncols = len(all_tts)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, tt in zip(axes, all_tts):
        for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
            grp = grp.sort_values("task_index")
            q_vals = grp["q_all_post"].apply(
                lambda d: d.get(tt) if isinstance(d, dict) else None
            )
            mask = q_vals.notna()
            if not mask.any():
                continue
            color = BEHAVIOR_COLORS.get(behavior, "#888")
            ax.plot(grp["task_index"][mask], q_vals[mask],
                    color=color, linewidth=_LW, alpha=0.7)

        ax.set_title(TASK_TYPE_LABELS.get(tt, f"TaskType {tt}"))
        ax.set_xlabel("Task")
        ax.set_ylabel("Q-value")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    _add_behavior_legend(axes[-1])
    fig.tight_layout()
    return fig


def plot_q_per_task_type_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std Q-value per behavior group, one subplot per task type."""
    rep = _drop_inactive(rep)
    if "q_all_post" not in rep.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "q_all_post not available — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    all_tts = sorted({
        tt
        for d in rep["q_all_post"].dropna()
        if isinstance(d, dict)
        for tt in d
        if tt in _TASK_TYPES_OF_INTEREST
    })
    if not all_tts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    rep = _with_round_zero(rep, {"q_all_post": {tt: 0.0 for tt in all_tts}})

    ncols = len(all_tts)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, tt in zip(axes, all_tts):
        col = rep["q_all_post"].apply(lambda d: d.get(tt) if isinstance(d, dict) else None)
        sub = rep.assign(_q=col).dropna(subset=["_q"])
        agg = sub.groupby(["behavior", "task_index"])["_q"].agg(["mean", "std"]).reset_index()
        for behavior, grp in agg.groupby("behavior"):
            grp = grp.sort_values("task_index")
            color = BEHAVIOR_COLORS.get(behavior, "#888")
            label_b = BEHAVIOR_LABELS.get(behavior, behavior)
            ax.plot(grp["task_index"], grp["mean"], color=color, linewidth=_LW, label=label_b)
            ax.fill_between(grp["task_index"],
                            grp["mean"] - grp["std"].fillna(0),
                            grp["mean"] + grp["std"].fillna(0),
                            alpha=_DEFAULT_ALPHA_FILL, color=color)
        ax.set_title(TASK_TYPE_LABELS.get(tt, f"TaskType {tt}"))
        ax.set_xlabel("Task")
        ax.set_ylabel("Q-value")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    axes[-1].legend(title="Behavior")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-task accuracy (from embedded run_data)
# ---------------------------------------------------------------------------

DATASET_COLORS = {
    "mnist":    "#2196F3",
    "cifar-10": "#FF9800",
    "cifar10":  "#FF9800",
}

def plot_accuracy_per_round_per_task(global_accuracy: pd.DataFrame) -> plt.Figure:
    """Line chart: global accuracy over rounds for every task, coloured by dataset.

    Uses the top-level ``global_accuracy`` table from the session pickle.
    """
    if global_accuracy.empty or "objective_global_accuracy" not in global_accuracy.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No global_accuracy data — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    fig, ax = plt.subplots(figsize=(12, 4))
    for (task_idx, dataset), grp in global_accuracy.groupby(["task_index", "dataset"]):
        grp = grp.sort_values("round")
        color = DATASET_COLORS.get(dataset.lower(), "#78909C")
        ax.plot(grp["round"], grp["objective_global_accuracy"],
                color=color, linewidth=1.4, alpha=0.6,
                label=f"T{task_idx} {dataset}" if task_idx < 6 else "_nolegend_")

    ax.set_xlabel("Round")
    ax.set_ylabel("Global accuracy")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    handles = [
        matplotlib.patches.Patch(facecolor=DATASET_COLORS.get(d.lower(), "#78909C"), label=d)
        for d in sorted({r for r in global_accuracy["dataset"].unique()})
    ]
    ax.legend(handles=handles, title="Dataset")
    fig.tight_layout()
    return fig


def plot_final_accuracy_per_task(global_accuracy: pd.DataFrame) -> plt.Figure:
    """Bar chart: final-round accuracy for every task, coloured by dataset.

    Uses the top-level ``global_accuracy`` table from the session pickle.
    """
    if global_accuracy.empty or "objective_global_accuracy" not in global_accuracy.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No global_accuracy data — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    final = (
        global_accuracy.sort_values("round")
        .groupby(["task_index", "dataset"], sort=False)
        .last()
        .reset_index()
    )
    all_indices = sorted(global_accuracy["task_index"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(all_indices) * 0.45), 4))
    for idx in all_indices:
        row = final[final["task_index"] == idx]
        if row.empty:
            ax.bar(idx, 0.02, color="#e0e0e0", edgecolor="#bdbdbd", linewidth=0.5)
        else:
            dataset = row["dataset"].iloc[0]
            acc     = float(row["objective_global_accuracy"].iloc[0])
            color   = DATASET_COLORS.get(dataset.lower(), "#78909C")
            ax.bar(idx, acc, color=color, edgecolor="black", linewidth=0.7)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    handles = [
        matplotlib.patches.Patch(facecolor=DATASET_COLORS.get(d.lower(), "#78909C"), label=d.upper())
        for d in sorted({r for r in global_accuracy["dataset"].unique()})
    ]
    ax.legend(handles=handles, title="Dataset", loc="upper left")
    fig.tight_layout()
    return fig


def plot_task_final_accuracy(tasks: list) -> plt.Figure:
    """Bar chart of each task's final-round global accuracy.

    All task slots are shown.  Tasks without logged accuracy data (empty logger
    due to REMOTE replay before bug-fix, or cached tasks) appear as light grey
    bars so the slot is visible but clearly marked as missing.
    """
    if not tasks:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No tasks", ha="center", va="center", transform=ax.transAxes)
        return fig

    all_indices = [t["task_index"] for t in tasks]
    all_datasets = [t.get("dataset", "").lower() for t in tasks]

    accuracies = []
    for t in tasks:
        rd = t.get("run_data")
        acc = None
        if rd is not None:
            global_df = rd.get("global", pd.DataFrame())
            if not global_df.empty and "objective_global_accuracy" in global_df.columns:
                vals = global_df["objective_global_accuracy"].dropna()
                if len(vals):
                    acc = float(vals.iloc[-1])
        accuracies.append(acc)

    has_any = any(a is not None for a in accuracies)
    fig, ax = plt.subplots(figsize=(max(8, len(all_indices) * 0.45), 4))

    for idx, dataset, acc in zip(all_indices, all_datasets, accuracies):
        if acc is None:
            ax.bar(idx, 0.02, color="#e0e0e0", edgecolor="#bdbdbd", linewidth=0.5)
        else:
            color = DATASET_COLORS.get(dataset, "#78909C")
            ax.bar(idx, acc, color=color, edgecolor="black", linewidth=0.7)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    if not has_any:
        ax.text(0.5, 0.5, "No accuracy data yet — re-run to populate",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888888")

    # Legend for datasets + missing marker
    seen = set()
    handles = []
    for dataset in all_datasets:
        if dataset not in seen:
            seen.add(dataset)
            color = DATASET_COLORS.get(dataset, "#78909C")
            handles.append(matplotlib.patches.Patch(facecolor=color, edgecolor="black", label=dataset.upper()))
    handles.append(matplotlib.patches.Patch(facecolor="#e0e0e0", edgecolor="#bdbdbd", label="no data"))
    ax.legend(handles=handles, title="Dataset", loc="upper left")

    fig.tight_layout()
    return fig


def plot_task_accuracy_curves(tasks: list) -> plt.Figure:
    """Line chart of global accuracy over rounds for each task (overlaid)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = plt.get_cmap("tab20")

    for idx, t in enumerate(tasks):
        rd = t.get("run_data")
        if rd is None:
            continue
        global_df = rd.get("global", pd.DataFrame())
        if global_df.empty or "objective_global_accuracy" not in global_df.columns:
            continue
        color = cmap(idx % 20)
        label = f"T{t['task_index']} ({t.get('dataset', '')})"
        ax.plot(global_df.index, global_df["objective_global_accuracy"],
                color=color, linewidth=1.2, alpha=0.7, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Global accuracy")
    ax.legend(title="Task", fontsize=7, ncol=4, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_behavior_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=color, linewidth=_LW)
        for color in BEHAVIOR_COLORS.values()
    ]
    labels = list(BEHAVIOR_LABELS.values())
    ax.legend(handles, labels, title="Behavior", fontsize=8)
