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
    "inactive":  "#7f7f7f",
}
BEHAVIOR_LABELS = {
    "honest":    "Honest",
    "malicious": "Malicious",
    "freerider": "Freerider",
    "inactive":  "Inactive",
}

_DEFAULT_ALPHA_FILL = 0.15
_LW = 2


def _user_color(name: str, behavior: str) -> str:
    base = BEHAVIOR_COLORS.get(behavior, "#888888")
    return base


# ---------------------------------------------------------------------------
# Per-user reputation evolution over tasks
# ---------------------------------------------------------------------------

def plot_tr_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Task Reputation (TR) for every user over task index."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["tr_post"], color=color, linewidth=_LW, alpha=0.7,
                label=f"{name} ({BEHAVIOR_LABELS.get(behavior, behavior)})")

    ax.set_xlabel("Task index")
    ax.set_ylabel("Task Reputation (TR)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_gir_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Global Integrity Reputation (GIR) for every user over task index."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["gir_post"], color=color, linewidth=_LW, alpha=0.7)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Global Integrity Reputation (GIR)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_q_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Q-value (selection pressure) for every user over task index."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["q_post"], color=color, linewidth=_LW, alpha=0.7)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Q-value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_balance_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """ETH balance for every user over task index."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["balance_post"], color=color, linewidth=_LW, alpha=0.7)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Balance (ETH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confidence_over_tasks(rep: pd.DataFrame) -> plt.Figure:
    """Confidence score for users with participation history, over task index.

    Only tasks where confidence data was recorded are shown (non-cached tasks
    and cached tasks after the first non-cached task for each user).
    Users with k=0 across all tasks are omitted (no history yet).
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    has_history = rep[rep["confidence"].notna() & rep["k"].notna() & (rep["k"] > 0)]
    for (name, behavior), grp in has_history.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["confidence"], color=color, linewidth=_LW, alpha=0.7)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Confidence")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 1.05)
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Group-level (mean ± std) evolution
# ---------------------------------------------------------------------------

def plot_tr_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std TR per behavior group over task index."""
    return _plot_metric_by_behavior(rep, "tr_post", "Task Reputation (TR)")


def plot_gir_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std GIR per behavior group over task index."""
    return _plot_metric_by_behavior(rep, "gir_post", "Global Integrity Reputation (GIR)")


def plot_q_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std Q-value per behavior group over task index."""
    return _plot_metric_by_behavior(rep, "q_post", "Q-value")


def plot_balance_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std balance per behavior group over task index."""
    return _plot_metric_by_behavior(rep, "balance_post", "Balance (ETH)")


def _plot_metric_by_behavior(rep: pd.DataFrame, col: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
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
        ax.fill_between(
            grp["task_index"],
            grp["mean"] - grp["std"].fillna(0),
            grp["mean"] + grp["std"].fillna(0),
            alpha=_DEFAULT_ALPHA_FILL, color=color,
        )

    ax.set_xlabel("Task index")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Behavior")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Selection analysis
# ---------------------------------------------------------------------------

def plot_selection_frequency(rep: pd.DataFrame) -> plt.Figure:
    """Bar chart: how many tasks each user was selected for."""
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
    fig, ax = plt.subplots(figsize=(11, 4))
    for (name, behavior), grp in rep.groupby(["user_name", "behavior"]):
        grp = grp.sort_values("task_index")
        color = BEHAVIOR_COLORS.get(behavior, "#888")
        ax.plot(grp["task_index"], grp["selection_score"], color=color, linewidth=_LW, alpha=0.5)
        sel = grp[grp["was_selected"]]
        ax.scatter(sel["task_index"], sel["selection_score"], color=color, s=25, zorder=5)

    ax.set_xlabel("Task index")
    ax.set_ylabel("Selection score")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _add_behavior_legend(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_vs_tr(rep: pd.DataFrame) -> plt.Figure:
    """Scatter: selection_score vs tr_pre, coloured by behavior."""
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


def plot_tr_per_task_type(rep: pd.DataFrame) -> plt.Figure:
    """One subplot per known task type showing each user's TR over tasks.

    Requires the ``tr_all_post`` column (dict keyed by task-type int).
    Falls back to a message if the column is absent (old session format).
    """
    if "tr_all_post" not in rep.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "tr_all_post not available — re-run to populate",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        return fig

    # Collect all task types present across the session
    all_tts = sorted({
        tt
        for d in rep["tr_all_post"].dropna()
        if isinstance(d, dict)
        for tt in d
    })
    if not all_tts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No TR data in tr_all_post", ha="center", va="center", transform=ax.transAxes)
        return fig

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
        ax.set_xlabel("Task index")
        ax.set_ylabel("Task Reputation (TR)")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

    _add_behavior_legend(axes[-1])
    fig.tight_layout()
    return fig


def plot_tr_per_task_type_by_behavior(rep: pd.DataFrame) -> plt.Figure:
    """Mean ± std TR per behavior group, one subplot per task type."""
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
    })
    if not all_tts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

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
        ax.set_xlabel("Task index")
        ax.set_ylabel("Task Reputation (TR)")
        ax.set_ylim(0, 1.05)
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
