import pickle
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"figure.dpi": 200})

ROLE_LABELS = {
    "good": "Honest",
    "bad": "Malicious",
    "freerider": "Freerider",
    "inactive": "Inactive",
}

BEHAVIOR_COLORS = {
    "good":      "#2196F3",
    "bad":       "#d62728",
    "freerider": "#9467bd",
    "inactive":  "#7f7f7f",
}

STRATEGY_COLORS = {
    "dotproduct":    "#2196F3",
    "naive":         "#FF9800",
    "accuracy_loss": "#E91E63",
    "accuracy_only": "#4CAF50",
    "loss_only":     "#9C27B0",
}


def plot_accuracy_loss_over_rounds(agg_global: pd.DataFrame) -> plt.Figure:
    """
    Dual-axis line chart: accuracy (left y-axis) + loss (right y-axis)
    with ±1 std shading.

    Expects columns: round, accuracy_mean, accuracy_std, loss_mean, loss_std.
    """
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    rounds = agg_global["round"]

    # Accuracy
    ax1.plot(rounds, agg_global["accuracy_mean"], color="#2196F3",
             linewidth=2, label="Accuracy")
    if "accuracy_std" in agg_global.columns:
        ax1.fill_between(
            rounds,
            agg_global["accuracy_mean"] - agg_global["accuracy_std"],
            agg_global["accuracy_mean"] + agg_global["accuracy_std"],
            alpha=0.2, color="#2196F3",
        )

    # Loss
    ax2.plot(rounds, agg_global["loss_mean"], color="#FF5722",
             linewidth=2, linestyle="--", label="Loss")
    if "loss_std" in agg_global.columns:
        ax2.fill_between(
            rounds,
            agg_global["loss_mean"] - agg_global["loss_std"],
            agg_global["loss_mean"] + agg_global["loss_std"],
            alpha=0.2, color="#FF5722",
        )

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Global Accuracy", color="#2196F3") # TODO: Check values
    ax2.set_ylabel("Global Loss", color="#FF5722") # TODO: Check values
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_strategy_comparison_lines(agg_by_strategy: pd.DataFrame) -> plt.Figure:
    """
    One line per strategy, mean accuracy over rounds with ±1 std error bands.

    Expects columns: contribution_score_strategy, round, accuracy_mean,
    accuracy_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for strategy, group in agg_by_strategy.groupby("contribution_score_strategy"):
        color = STRATEGY_COLORS.get(strategy, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["accuracy_mean"],
                label=strategy, color=color, linewidth=2)
        if "accuracy_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["accuracy_mean"] - group["accuracy_std"],
                group["accuracy_mean"] + group["accuracy_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_strategy_comparison_boxplot(agg_final: pd.DataFrame) -> plt.Figure:
    """
    One box per strategy showing final-round accuracy distribution.

    Expects columns: contribution_score_strategy, final_accuracy.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    strategies = sorted(agg_final["contribution_score_strategy"].unique())
    data = [
        agg_final.loc[
            agg_final["contribution_score_strategy"] == s, "final_accuracy"
        ].values
        for s in strategies
    ]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strategies]

    bp = ax.boxplot(data, patch_artist=True, labels=strategies)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Final-Round Accuracy (%)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_grs_by_role(agg_grs: pd.DataFrame) -> plt.Figure:
    """
    One line per role (eventual user type), GRS over rounds with ±1 std shading.

    Expects columns: role, round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_grs.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "grs_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["grs_mean"] - group["grs_std"],
                group["grs_mean"] + group["grs_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_contribution_score_by_role(agg_scores: pd.DataFrame) -> plt.Figure:
    """
    One line per role (eventual user type), contribution score over rounds
    with ±1 std shading.

    Expects columns: role, round, score_mean, score_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_scores.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["score_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "score_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["score_mean"] - group["score_std"],
                group["score_mean"] + group["score_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Contribution Score")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grs_by_role_relative(agg_grs: pd.DataFrame) -> plt.Figure:
    """
    One line per role, GRS over rounds-since-activation with ±1 std shading.
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, grs_mean, grs_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_grs.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("relative_round")
        ax.plot(group["relative_round"], group["grs_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "grs_std" in group.columns:
            ax.fill_between(
                group["relative_round"],
                group["grs_mean"] - group["grs_std"],
                group["grs_mean"] + group["grs_std"],
                alpha=0.15, color=color,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Activation")
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_contribution_score_by_role_relative(agg_scores: pd.DataFrame) -> plt.Figure:
    """
    One line per role, contribution score over rounds-since-activation with ±1 std shading.
    A vertical dashed line at x=0 marks the activation moment.

    Expects columns: role, relative_round, score_mean, score_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for role, group in agg_scores.groupby("role"):
        color = BEHAVIOR_COLORS.get(role, None)
        group = group.sort_values("relative_round")
        ax.plot(group["relative_round"], group["score_mean"],
                label=ROLE_LABELS[role], color=color, linewidth=2)
        if "score_std" in group.columns:
            ax.fill_between(
                group["relative_round"],
                group["score_mean"] - group["score_std"],
                group["score_mean"] + group["score_std"],
                alpha=0.15, color=color,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Activation")
    ax.set_xlabel("Rounds since activation")
    ax.set_ylabel("Contribution Score")
    ax.legend(title="Role")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grs_by_user(grs_users: pd.DataFrame) -> plt.Figure:
    # x-axis: Round: 0,1,...,n
    # y-axis: GRS
    # Plot user as a line
    fig, ax = plt.subplots(figsize=(9, 4))

    # for user_id, group in grs_users.groupby("user_id, behavior"):
    #     ax.plot(group["round"], group["grs"], label=f"User {user_id}", alpha=0.5)

    for (user_id, behavior), group in grs_users.groupby(["user_id", "role"]):
        ax.plot(group["round"], group["grs"], label=f"User {user_id} ({ROLE_LABELS[behavior]})", alpha=0.5) # alpha: 50% transparency, so overlapping lines show through each other

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Reputation Score (ETH)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Users")
    ax.grid(True, alpha=0.3) # alpha: makes the grid subtle/faint so it doesn't compete with the data
    fig.tight_layout()
    return fig


def plot_global_acc_by_aggregation_strategy(acc_by_strategy: pd.DataFrame) -> plt.Figure:
    """
    One line per aggregation rule, mean accuracy over rounds with ±1 std shading.

    Expects columns: aggregation_rule, round, accuracy_mean, accuracy_std.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy, group in acc_by_strategy.groupby("aggregation_rule"):
        color = STRATEGY_COLORS.get(strategy)
        group = group.sort_values("round")
        ax.plot(group["round"], group["accuracy_mean"], label=strategy, color=color, linewidth=2)
        if "accuracy_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["accuracy_mean"] - group["accuracy_std"],
                group["accuracy_mean"] + group["accuracy_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy") # TODO: Not a percentage
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Agg. Strategy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig



def plot_global_loss_by_aggregation_strategy(loss_by_strategy: pd.DataFrame) -> plt.Figure:
    """
    One line per aggregation rule, mean loss over rounds with ±1 std shading.

    Expects columns: aggregation_rule, round, loss_mean, loss_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for strategy, group in loss_by_strategy.groupby("aggregation_rule"):
        color = STRATEGY_COLORS.get(strategy)
        group = group.sort_values("round")
        ax.plot(group["round"], group["loss_mean"], label=strategy, color=color, linewidth=2)
        if "loss_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["loss_mean"] - group["loss_std"],
                group["loss_mean"] + group["loss_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Global Loss") # TODO: Not a percentage
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Agg. Strategy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig




def plot_gas_cost_by_tx_type(agg_gas: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart of mean gas used per transaction type, one bar group per
    tx_type and one bar per contribution_score_strategy, with ±1 std error bars.

    Expects columns: tx_type, contribution_score_strategy, gas_mean, gas_std.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    tx_types = sorted(agg_gas["tx_type"].unique())
    strategies = sorted(agg_gas["contribution_score_strategy"].unique())
    n_tx = len(tx_types)
    n_strategies = len(strategies)
    width = 0.8 / n_strategies
    x = range(n_tx)

    for i, strategy in enumerate(strategies):
        group = agg_gas[agg_gas["contribution_score_strategy"] == strategy]
        means = []
        stds = []
        for tx in tx_types:
            row = group[group["tx_type"] == tx]
            means.append(row["gas_mean"].iloc[0] if not row.empty else float("nan"))
            stds.append(row["gas_std"].iloc[0] if not row.empty else 0)

        xpos = [xi - 0.4 + i * width + width / 2 for xi in x]
        color = STRATEGY_COLORS.get(strategy, "#607c8a")
        ax.bar(xpos, means, width, yerr=stds, capsize=4,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.8,
               label=strategy)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tx_types, rotation=10, ha="right")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Mean Gas Used")
    ax.legend(title="Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig



def plot_round_kicked_by_strategy(
    agg_kicked: pd.DataFrame,
    title: str = "Effectiveness of Strategies in Removing Dishonest Participants",
    max_rounds: int | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: for each contribution score strategy, show at which
    round each user role was disqualified (lower = removed sooner = better).
    Asymmetric error bars show min/max range across runs.

    Inspired by kickedGraph() in scripts/processData.py.

    Expects columns: contribution_score_strategy, role,
                     mean_round_kicked, low_err, high_err.
    """
    if agg_kicked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No disqualified users", ha="center", va="center", transform=ax.transAxes)
        return fig

    strategies = sorted(agg_kicked["contribution_score_strategy"].unique())
    roles = sorted(agg_kicked["role"].unique())

    n_strategies = len(strategies)
    n_roles = len(roles)
    x = range(n_strategies)
    width = 0.8 / n_roles

    fig, ax = plt.subplots(figsize=(max(7, n_strategies * 1.8), 5))

    for role_idx, role in enumerate(roles):
        role_data = agg_kicked[agg_kicked["role"] == role]
        color = BEHAVIOR_COLORS.get(role, "#888888")

        means   = []
        low_err = []
        high_err = []
        missing = []

        for strategy in strategies:
            row = role_data[role_data["contribution_score_strategy"] == strategy]
            if row.empty:
                means.append(float("nan"))
                low_err.append(0)
                high_err.append(0)
                missing.append(True)
            else:
                means.append(row["mean_round_kicked"].iloc[0])
                low_err.append(row["low_err"].iloc[0])
                high_err.append(row["high_err"].iloc[0])
                missing.append(False)

        xpos = [xi - 0.4 + role_idx * width + width / 2 for xi in x]

        bar_means = [m if not missing[i] else float("nan") for i, m in enumerate(means)]
        show_err = any(l != 0 or h != 0 for l, h in zip(low_err, high_err))

        ax.bar(
            xpos,
            bar_means,
            width,
            yerr=[low_err, high_err] if show_err else None,
            capsize=4,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            label=ROLE_LABELS[role],
        )

        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (max_rounds or 1)
        for xi, is_missing in zip(xpos, missing):
            if is_missing:
                ax.text(
                    xi, y_top * 0.02, "N/A",
                    ha="center", va="bottom",
                    fontsize=8, color="gray", rotation=90,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=10, ha="right")
    ax.set_ylabel("Round Kicked (lower = removed sooner)")
    ax.set_title(title)
    ax.legend(title="Role")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig



def plot_merge_weights_by_behavior(agg_weights: pd.DataFrame, stats: pd.DataFrame | None = None) -> plt.Figure:
    """
    One line per behavior, average merge weight over rounds with ±1 std shading.
    Rounds where a behavior was never merged will have no point (NaN weight_mean).

    Expects agg_weights columns: behavior, round, weight_mean, weight_std.
    Expects stats columns: behavior, total_rounds, rounds_merged, pct_merged, users_merged.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    for behavior, group in agg_weights.groupby("behavior"):
        color = BEHAVIOR_COLORS.get(behavior, None)
        group = group.sort_values("round")
        ax.plot(group["round"], group["weight_mean"],
                label=ROLE_LABELS.get(behavior, behavior), color=color, linewidth=2)
        if "weight_std" in group.columns:
            ax.fill_between(
                group["round"],
                group["weight_mean"] - group["weight_std"],
                group["weight_mean"] + group["weight_std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Round")
    ax.set_ylabel("Merge Weight")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    if stats is not None:
        stats_by_behavior = stats.set_index("behavior")
        handles, labels = [], []
        for behavior in agg_weights["behavior"].unique():
            color = BEHAVIOR_COLORS.get(behavior, "black")
            role  = ROLE_LABELS.get(behavior, behavior)
            handle = Line2D([0], [0], color=color, linewidth=2)
            if behavior in stats_by_behavior.index:
                row    = stats_by_behavior.loc[behavior]
                merged = int(row["rounds_merged"]) if pd.notna(row["rounds_merged"]) else 0
                total  = int(row["total_rounds"])
                users  = int(row["user_count"])
                label  = f"{role:<14}  {merged:>2}/{total:<2} rounds  ·  {users} user(s)"
            else:
                label = role
            handles.append(handle)
            labels.append(label)
        ax.legend(
            handles, labels,
            title="Not-merged by behavior",
            loc="lower right",
            fontsize=8,
            prop={"family": "monospace", "size": 8},
            framealpha=0.9,
            edgecolor="#cccccc",
        )
    else:
        ax.legend(title="Behavior")
    fig.tight_layout()
    return fig



FIGURE_FORMATS = ["png"]


def set_figure_format(fmt: str | list[str]):
    """Set the output format(s). Accepts a single format or a comma-separated
    string / list of formats (e.g. "svg,pdf")."""
    global FIGURE_FORMATS
    if isinstance(fmt, str):
        fmt = [f.strip() for f in fmt.split(",") if f.strip()]
    FIGURE_FORMATS = fmt or ["png"]


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s/]+", "-", text).strip("-")


def _panel_groups(fig: plt.Figure) -> list[dict]:
    """Group fig.axes into independent side-by-side panels.

    Axes whose horizontal extents overlap (e.g. a twin axes sharing the same
    plot area) are merged into one group. Colorbar axes are attached to the
    group they sit immediately to the right of.
    """
    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    main_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]

    groups: list[dict] = []
    for ax in main_axes:
        bbox = ax.get_position()
        for g in groups:
            gb = g["bbox"]
            if bbox.x0 < gb.x1 and bbox.x1 > gb.x0:
                g["axes"].append(ax)
                g["bbox"] = Bbox.from_extents(
                    min(gb.x0, bbox.x0), min(gb.y0, bbox.y0),
                    max(gb.x1, bbox.x1), max(gb.y1, bbox.y1))
                break
        else:
            groups.append({"bbox": bbox, "axes": [ax]})

    for cax in cbar_axes:
        cb = cax.get_position()
        nearest = min(groups, key=lambda g: abs(g["bbox"].x1 - cb.x0))
        nearest["axes"].append(cax)

    groups.sort(key=lambda g: g["bbox"].x0)
    return groups


def save_figure_panels(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """If `fig` has more than one side-by-side panel, also save each panel as
    its own file: ``<stem>_panel<N>[-<title>]<suffix>`` next to `path`.

    No-op for single-panel figures.
    """
    groups = _panel_groups(fig)
    if len(groups) < 2:
        return

    suptitle = fig.get_suptitle()
    fallback_ylabel = next(
        (ax.get_ylabel() for ax in fig.axes
         if ax.get_label() != "<colorbar>" and ax.get_ylabel()), "")
    orig_w, orig_h = fig.get_size_inches()
    panel_w = orig_w / len(groups)
    index_of = {ax: i for i, ax in enumerate(fig.axes)}

    for n, g in enumerate(groups, start=1):
        fig2 = pickle.loads(pickle.dumps(fig))
        keep = {index_of[ax] for ax in g["axes"]}
        for i, ax in enumerate(list(fig2.axes)):
            if i not in keep:
                ax.remove()

        cbar2 = [ax for ax in fig2.axes if ax.get_label() == "<colorbar>"]
        main2 = [ax for ax in fig2.axes if ax.get_label() != "<colorbar>"]

        for ax in main2:
            ax.yaxis.set_tick_params(labelleft=True)
            for lbl in ax.get_yticklabels():
                lbl.set_visible(True)
            if not cbar2 and not ax.get_ylabel() and fallback_ylabel:
                ax.set_ylabel(fallback_ylabel)

        if cbar2:
            for ax in main2:
                ax.set_position([0.1, 0.11, 0.72, 0.78])
            for ax in cbar2:
                ax.set_position([0.86, 0.11, 0.03, 0.78])
            fig2.set_size_inches(panel_w * 1.2, orig_h)
        else:
            for ax in main2:
                ax.set_position([0.13, 0.12, 0.82, 0.76])
            fig2.set_size_inches(panel_w, orig_h)

        title = main2[0].get_title() if main2 else ""
        if not title and suptitle and main2:
            title = suptitle
            main2[0].set_title(suptitle, fontsize=10)

        if fig2._suptitle is not None:
            fig2._suptitle.remove()

        slug = _slugify(title)
        name = f"panel{n}-{slug}" if slug else f"panel{n}"
        out = path.parent / f"{path.stem}_{name}{path.suffix}"
        fig2.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)


def _format_path(path: Path, fmt: str, base_root: Path | None) -> Path:
    """Path for *fmt*, rooted under a format-specific sibling of *base_root*
    (e.g. ``figures/aggregate`` -> ``figures/aggregate_svg``). Falls back to
    *path* unchanged if no base_root is given, for the "png" format, or if
    *path* is not under *base_root*."""
    path = path.with_suffix(f".{fmt}")
    if base_root is None or fmt == "png":
        return path
    try:
        rel = path.relative_to(base_root)
    except ValueError:
        return path
    return base_root.with_name(f"{base_root.name}_{fmt}") / rel


def save_figure(fig: plt.Figure, path, dpi: int = 150, base_root: Path | None = None):
    """Save *fig* in every format set via :func:`set_figure_format`.

    If *base_root* is given and multiple formats (or a non-png format) are
    requested, each format is written under a sibling directory of
    *base_root* named ``<base_root>_<fmt>`` (png stays under *base_root*),
    so the figure is computed once but saved everywhere it's needed.
    """
    path = Path(path)
    for fmt in FIGURE_FORMATS:
        out_path = _format_path(path, fmt, base_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        save_figure_panels(fig, out_path, dpi=dpi)
