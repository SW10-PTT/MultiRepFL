"""Plot functions for multirep session data.

All functions accept a ``reputation_timeline`` DataFrame (as produced by
MultirepLogger) and/or ``tasks`` list.  They return matplotlib Figure objects
so the caller can save them with ``plots.save_figure``.
"""

from pathlib import Path

import matplotlib
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
    """Confidence score for every user who was selected, over task index."""
    fig, ax = plt.subplots(figsize=(11, 4))
    selected = rep[rep["was_selected"] & rep["confidence"].notna()]
    for (name, behavior), grp in selected.groupby(["user_name", "behavior"]):
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
# Per-task accuracy (from embedded run_data)
# ---------------------------------------------------------------------------

def plot_task_final_accuracy(tasks: list) -> plt.Figure:
    """Bar chart of each task's final-round global accuracy from embedded run_data."""
    indices, accuracies, datasets = [], [], []
    for t in tasks:
        rd = t.get("run_data")
        if rd is None:
            continue
        global_df = rd.get("global", pd.DataFrame())
        if global_df.empty or "objective_global_accuracy" not in global_df.columns:
            continue
        final_acc = global_df["objective_global_accuracy"].dropna().iloc[-1] if not global_df.empty else None
        if final_acc is None:
            continue
        indices.append(t["task_index"])
        accuracies.append(float(final_acc))
        datasets.append(t.get("dataset", ""))

    if not indices:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No accuracy data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(6, len(indices) * 0.5), 4))
    ax.bar(indices, accuracies, color="#2196F3", edgecolor="black", linewidth=0.7)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Final global accuracy")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
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
