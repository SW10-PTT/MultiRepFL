"""Generate aggregate comparison graphs across multiple multirep experiments.

Scans a root folder (default: experiment/data/FinishedRuns) where each
sub-directory is one experiment containing one or more run tarballs.  Pairs the
globalrep and multirep variants of each experiment, averages their runs, and
writes comparison PNGs under figures/aggregate/<pair>/.

Usage:
    python analysis/multirep_aggregate_graphs.py
    python analysis/multirep_aggregate_graphs.py --root <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from analysis import multirep_aggregate_plots as ap
from analysis import multirep_grouped_plots as gp
from analysis import multirep_plots as mrp
from analysis import multirep_thesis_plots as tp
from analysis.multirep_runavg import averaged_views
from analysis.multirep_aggregate_loader import (
    CIFAR_TT,
    MNIST_TT,
    ExperimentPair,
    ExperimentRuns,
    build_pairs,
    discover_experiments,
    find_experiment,
)
from analysis.plots import save_figure as _save

def save_figure(fig, path):
    """Save and immediately close the figure to keep memory bounded."""
    _save(fig, path)
    plt.close(fig)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "experiment" / "data" / "FinishedRuns"
DEFAULT_OUT = REPO_ROOT / "figures" / "aggregate"

TASK_TYPES = [MNIST_TT, CIFAR_TT]
DS_NAME = {MNIST_TT: "mnist", CIFAR_TT: "cifar10"}


def _generate_pair_graphs(pair: ExperimentPair, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0

    # --- 1. accuracy & loss over rounds (full + thirds), per dataset ---
    for tt in TASK_TYPES:
        ds = DS_NAME[tt]
        for method in ("mean", "median"):
            save_figure(
                ap.plot_metric_over_rounds(pair, tt, "objective_global_accuracy",
                                           "Global accuracy", method),
                out_dir / f"accuracy_{ds}_{method}_full.png")
            save_figure(
                ap.plot_metric_thirds(pair, tt, "objective_global_accuracy",
                                      "Global accuracy", method),
                out_dir / f"accuracy_{ds}_{method}_thirds.png")
            n += 2
        # loss (mean only)
        save_figure(
            ap.plot_metric_over_rounds(pair, tt, "objective_global_loss",
                                       "Global loss", "mean"),
            out_dir / f"loss_{ds}_full.png")
        save_figure(
            ap.plot_metric_thirds(pair, tt, "objective_global_loss",
                                  "Global loss", "mean"),
            out_dir / f"loss_{ds}_thirds.png")
        n += 2

    # --- 2. final accuracy & time-to-accuracy ---
    save_figure(ap.plot_final_accuracy(pair), out_dir / "final_accuracy.png")
    save_figure(ap.plot_time_to_accuracy(pair, "threshold"), out_dir / "time_to_accuracy_threshold.png")
    save_figure(ap.plot_time_to_accuracy(pair, "fraction"), out_dir / "time_to_accuracy_fraction.png")
    n += 3

    # --- 3. selection rate by behavior, per dataset ---
    for tt in TASK_TYPES:
        save_figure(ap.plot_selection_rate_by_behavior(pair, tt),
                    out_dir / f"selection_rate_{DS_NAME[tt]}.png")
        n += 1

    # --- 4. TR / GIR development + selection over time ---
    save_figure(ap.plot_tr_development(pair), out_dir / "tr_development.png")
    save_figure(ap.plot_gir_development(pair), out_dir / "gir_development.png")
    save_figure(ap.plot_selection_rate_over_time(pair), out_dir / "selection_rate_over_time.png")
    n += 3

    # --- 5. cold-start selection on CIFAR ---
    save_figure(ap.plot_cold_start_selection(pair), out_dir / "cold_start_cifar_selection.png")
    n += 1

    # --- 6. kicked ---
    save_figure(ap.plot_kicked_round(pair), out_dir / "kicked_round.png")
    save_figure(ap.plot_kicked_rate(pair), out_dir / "kicked_rate.png")
    n += 2

    # --- 7. grouped by data-split category + behavior ---
    for tt in TASK_TYPES:
        save_figure(gp.plot_selection_rate_by_split(pair, tt),
                    out_dir / f"split_selection_{DS_NAME[tt]}.png")
        save_figure(gp.plot_tr_by_split(pair, tt),
                    out_dir / f"split_tr_{DS_NAME[tt]}.png")
        save_figure(gp.plot_final_accuracy_by_dominant_split(pair, tt),
                    out_dir / f"split_final_accuracy_{DS_NAME[tt]}.png")
        n += 3
    save_figure(gp.plot_gir_by_split(pair), out_dir / "split_gir.png")
    save_figure(gp.plot_final_balance_by_split(pair), out_dir / "split_final_balance.png")
    save_figure(gp.plot_selection_propensity_by_split(pair), out_dir / "split_selection_propensity.png")
    n += 3

    # --- 8. proposed thesis graphs ---
    save_figure(tp.plot_score_decomposition(pair), out_dir / "score_decomposition.png")
    save_figure(tp.plot_cold_start_latency(pair, CIFAR_TT), out_dir / "cold_start_latency_cifar.png")
    save_figure(tp.plot_run_variability(pair), out_dir / "run_variability.png")
    save_figure(tp.plot_freerider_economics(pair), out_dir / "freerider_economics.png")
    n += 4
    for tt in TASK_TYPES:
        save_figure(tp.plot_taskrep_accuracy_correlation(pair, tt),
                    out_dir / f"taskrep_accuracy_corr_{DS_NAME[tt]}.png")
        save_figure(tp.plot_selection_efficiency(pair, tt),
                    out_dir / f"selection_efficiency_{DS_NAME[tt]}.png")
        n += 2

    # --- 9. run-averaged ports of the single-run graphs, per system ---
    for system, exp in pair.items():
        n += _generate_runavg_graphs(exp, out_dir / f"runavg-{system}")

    return n


def _generate_runavg_graphs(exp, out_dir: Path) -> int:
    """Run-averaged versions of the original single-session multirep graphs."""
    rep, ga = averaged_views(exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if not rep.empty:
        for fn, name in [
            (mrp.plot_tr_per_task_type, "tr_per_task_type"),
            (mrp.plot_tr_per_task_type_by_behavior, "tr_per_task_type_by_behavior"),
            (mrp.plot_q_per_task_type, "q_per_task_type"),
            (mrp.plot_q_per_task_type_by_behavior, "q_per_task_type_by_behavior"),
            (mrp.plot_tr_over_tasks, "tr_per_user"),
            (mrp.plot_gir_over_tasks, "gir_per_user"),
            (mrp.plot_balance_over_tasks, "balance_per_user"),
            (mrp.plot_confidence_over_tasks, "confidence_per_user"),
            (mrp.plot_tr_by_behavior, "tr_by_behavior"),
            (mrp.plot_gir_by_behavior, "gir_by_behavior"),
            (mrp.plot_balance_by_behavior, "balance_by_behavior"),
            (mrp.plot_selection_frequency, "selection_frequency"),
            (mrp.plot_selection_heatmap, "selection_heatmap"),
            (mrp.plot_selection_score_over_tasks, "selection_score"),
            (mrp.plot_score_vs_tr, "score_vs_tr"),
        ]:
            save_figure(fn(rep), out_dir / f"{name}.png")
            n += 1
    if not ga.empty:
        fig = mrp.plot_accuracy_per_round_per_task(ga)
        if fig.axes:
            fig.axes[0].set_xlim(0, 15)  # honour the round cap on the overview
        save_figure(fig, out_dir / "task_accuracy_curves.png")
        save_figure(mrp.plot_final_accuracy_per_task(ga), out_dir / "task_final_accuracy.png")
        n += 2
    return n


def _generate_qvalue_graphs(experiments: list[ExperimentRuns], out_dir: Path) -> int:
    """Special comparison: multirep avg-distribution (with Q) vs noqvalue (without Q)."""
    without_q = find_experiment(experiments, "multirep", "noqvalue", "avg-distribution")
    # the plain avg-distribution multirep, excluding the noqvalue variant
    with_q = None
    for exp in experiments:
        low = exp.name.lower()
        if "multirep" in low and "avg-distribution" in low and "noqvalue" not in low:
            with_q = exp
            break
    if with_q is None or without_q is None:
        print("  [info] q-value comparison skipped (need both multirep avg & noqvalue-avg).")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(ap.plot_qvalue_selection_wait(with_q, without_q),
                out_dir / "qvalue_selection_wait.png")
    print(f"  q-value comparison: {with_q.name}  vs  {without_q.name}")
    return 1


def main(root: Path, out: Path) -> None:
    print(f"Scanning experiments under: {root}")
    experiments = discover_experiments(root)
    if not experiments:
        print("No experiments found.")
        return

    pairs = build_pairs(experiments)
    total = 0
    for pair in pairs:
        if not pair.is_complete():
            sides = ", ".join(s for s, _ in pair.items())
            print(f"  [skip] '{pair.label}' is not a complete pair (have: {sides})")
            continue
        pair_dir = out / pair.key.removeprefix("exp-")
        print(f"Pair '{pair.label}' → {pair_dir}")
        total += _generate_pair_graphs(pair, pair_dir)

    # Special q-value experiment comparison
    total += _generate_qvalue_graphs(experiments, out / "qvalue-comparison")

    print(f"\nDone. {total} graphs written under {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate aggregate multirep comparison graphs")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Root folder of experiments (default: experiment/data/FinishedRuns)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output folder (default: figures/aggregate)")
    args = parser.parse_args()
    main(args.root, args.out)
