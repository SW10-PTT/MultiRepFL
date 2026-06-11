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
from analysis import multirep_variant_compare as vc
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
from analysis.plots import save_figure as _save, set_figure_format

_OUT_BASE: Path | None = None


def save_figure(fig, path):
    """Save (in every requested format, under format-specific sibling
    directories of _OUT_BASE) and immediately close the figure to keep
    memory bounded."""
    _save(fig, path, base_root=_OUT_BASE)
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

    # --- 4b. single-dataset progression variants (no flat gaps) ---
    for tt in TASK_TYPES:
        ds = DS_NAME[tt]
        save_figure(ap.plot_metric_progression(pair, tt, "tr"), out_dir / f"tr_progression_{ds}.png")
        save_figure(ap.plot_metric_progression(pair, tt, "gir"), out_dir / f"gir_progression_{ds}.png")
        save_figure(ap.plot_metric_progression(pair, tt, "selection"),
                    out_dir / f"selection_progression_{ds}.png")
        save_figure(gp.plot_tr_by_split_progression(pair, tt), out_dir / f"split_tr_{ds}_only.png")
        n += 4

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
    save_figure(gp.plot_net_earnings_by_split(pair), out_dir / "split_net_earnings.png")
    save_figure(gp.plot_net_earnings_by_split_dataset(pair), out_dir / "split_net_earnings_by_dataset.png")
    save_figure(gp.plot_selection_propensity_by_split(pair), out_dir / "split_selection_propensity.png")
    save_figure(gp.plot_mixed_behavior_users(pair), out_dir / "mixed_behavior_users.png")
    n += 5

    # --- 7b. task-hopper development (self-skips unless mixed-behaviour users exist) ---
    save_figure(gp.plot_taskhopper_reputation_development(pair),
                out_dir / "taskhopper_reputation_development.png")
    save_figure(gp.plot_taskhopper_selection_development(pair),
                out_dir / "taskhopper_selection_development.png")
    save_figure(gp.plot_taskhopper_selection_development(pair, cumulative=False),
                out_dir / "taskhopper_selection_development_actual.png")
    n += 3

    # --- 7c. task-hopper-only: behaviour-role buckets + per-role/per-user dev ---
    if gp._taskhoppers_present(pair):
        # role-bucketed versions of the split graphs (Honest pooled; each
        # malicious / free-rider variant in its own bucket)
        # NB: no role version of final_accuracy_by_dominant_split — the dominant
        # selected role is always "Honest" (the majority every task), so it is
        # degenerate.  Role buckets are used for the participant-level splits only.
        for tt in TASK_TYPES:
            ds = DS_NAME[tt]
            save_figure(gp.plot_selection_rate_by_split(pair, tt, bucket_fn=gp.role_bucket),
                        out_dir / f"rolesplit_selection_{ds}.png")
            save_figure(gp.plot_tr_by_split(pair, tt, bucket_fn=gp.role_bucket),
                        out_dir / f"rolesplit_tr_{ds}.png")
            n += 2
        save_figure(gp.plot_gir_by_split(pair, bucket_fn=gp.role_bucket),
                    out_dir / "rolesplit_gir.png")
        save_figure(gp.plot_net_earnings_by_split(pair, bucket_fn=gp.role_bucket),
                    out_dir / "rolesplit_net_earnings.png")
        # mixed-behaviour earnings incl. honest baseline (req: add honest users)
        save_figure(gp.plot_mixed_behavior_users(pair, include_honest=True),
                    out_dir / "mixed_behavior_users_with_honest.png")
        # per-user / per-role reputation development + selections-by-type
        save_figure(gp.plot_mixed_behavior_tr_development(pair),
                    out_dir / "mixed_behavior_tr_development.png")
        save_figure(gp.plot_taskhopper_reputation_development_by_role(pair),
                    out_dir / "taskhopper_reputation_development_by_role.png")
        save_figure(gp.plot_selections_by_role_dataset(pair),
                    out_dir / "selections_by_role_dataset.png")
        save_figure(gp.plot_taskhopper_selection_development_by_role(pair),
                    out_dir / "taskhopper_selection_development_by_role.png")
        save_figure(gp.plot_taskhopper_selection_development_by_role(pair, cumulative=False),
                    out_dir / "taskhopper_selection_development_by_role_actual.png")
        save_figure(gp.plot_taskhopper_selection_development_individual(pair),
                    out_dir / "taskhopper_selection_development_individual.png")
        save_figure(gp.plot_taskhopper_selection_development_individual(pair, cumulative=False),
                    out_dir / "taskhopper_selection_development_individual_actual.png")
        save_figure(gp.plot_taskhopper_tr_development_by_role(pair),
                    out_dir / "taskhopper_tr_development_by_role.png")
        save_figure(gp.plot_taskhopper_tr_development_individual(pair),
                    out_dir / "taskhopper_tr_development_individual.png")
        n += 11
        # per-dataset versions of the selection / TR development graphs
        for tt in TASK_TYPES:
            ds = DS_NAME[tt]
            save_figure(gp.plot_taskhopper_selection_development_by_role(pair, task_type=tt),
                        out_dir / f"taskhopper_selection_development_by_role_{ds}.png")
            save_figure(gp.plot_taskhopper_selection_development_by_role(pair, cumulative=False, task_type=tt),
                        out_dir / f"taskhopper_selection_development_by_role_{ds}_actual.png")
            save_figure(gp.plot_taskhopper_selection_development_individual(pair, task_type=tt),
                        out_dir / f"taskhopper_selection_development_individual_{ds}.png")
            save_figure(gp.plot_taskhopper_selection_development_individual(pair, cumulative=False, task_type=tt),
                        out_dir / f"taskhopper_selection_development_individual_{ds}_actual.png")
            save_figure(gp.plot_taskhopper_tr_development_by_role_dataset(pair, tt),
                        out_dir / f"taskhopper_tr_development_by_role_{ds}.png")
            save_figure(gp.plot_taskhopper_tr_development_individual_dataset(pair, tt),
                        out_dir / f"taskhopper_tr_development_individual_{ds}.png")
            n += 6

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
        save_figure(tp.plot_contrib_vs_data_richness(pair, tt),
                    out_dir / f"contrib_vs_data_richness_{DS_NAME[tt]}.png")
        n += 3

    # --- 8b. new thesis graphs (specialisation, significance, adversarial) ---
    save_figure(tp.plot_tr_cross_task_transfer(pair), out_dir / "tr_cross_task_transfer.png")
    save_figure(tp.plot_specialization_heatmap(pair), out_dir / "specialization_heatmap.png")
    save_figure(tp.plot_final_tr_spread(pair), out_dir / "final_tr_spread.png")
    save_figure(tp.plot_final_accuracy_ci(pair), out_dir / "final_accuracy_ci.png")
    save_figure(tp.plot_selection_merit_spearman(pair), out_dir / "selection_merit_spearman.png")
    save_figure(tp.plot_cumulative_earnings_by_behavior(pair), out_dir / "cumulative_earnings_by_behavior.png")
    save_figure(tp.plot_detection_rate_over_time(pair), out_dir / "detection_rate_over_time.png")
    n += 7

    # --- 9. run-averaged ports of the single-run graphs, per system ---
    for system, exp in pair.items():
        n += _generate_runavg_graphs(exp, out_dir / f"runavg-{system}")

    return n


def _generate_runavg_graphs(exp, out_dir: Path) -> int:
    """Run-averaged versions of the original single-session multirep graphs."""
    rep, ga = averaged_views(exp)
    raw = exp.reputation_timeline()  # run-tagged, for selected-only progression
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if not raw.empty:
        # progression where the line advances only on tasks the user was selected for
        save_figure(mrp.plot_tr_selected_progression(raw), out_dir / "tr_selected_progression.png")
        save_figure(mrp.plot_gir_selected_progression(raw), out_dir / "gir_selected_progression.png")
        save_figure(mrp.plot_balance_selected_progression(raw), out_dir / "balance_selected_progression.png")
        n += 3
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
        # per-dataset selection frequency (MNIST tasks only / CIFAR tasks only)
        if "task_type" in rep.columns:
            for tt, ds in [(5, "mnist"), (6, "cifar10")]:
                sub = rep[rep["task_type"] == tt]
                if not sub.empty:
                    save_figure(mrp.plot_selection_frequency(sub),
                                out_dir / f"selection_frequency_{ds}.png")
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
    save_figure(ap.plot_qvalue_effect(with_q, without_q),
                out_dir / "qvalue_effect.png")
    save_figure(ap.plot_qvalue_selection_wait(with_q, without_q),
                out_dir / "qvalue_selection_wait.png")
    save_figure(ap.plot_qvalue_coverage(with_q, without_q),
                out_dir / "qvalue_coverage.png")
    save_figure(ap.plot_qvalue_mechanism(with_q, without_q),
                out_dir / "qvalue_mechanism.png")
    print(f"  q-value comparison: {with_q.name}  vs  {without_q.name}")
    return 1


def _generate_qslot_graphs(experiments: list[ExperimentRuns], out_dir: Path) -> int:
    """Special comparison: multirep task-hopper with the Q-slot cap (qslot2) vs
    the uncapped baseline.  Both are multi-rep, so the globalrep/multirep pairing
    does not pair them."""
    capped = find_experiment(experiments, "multirep", "task-hopper", "qslot2")
    baseline = None
    for exp in experiments:
        low = exp.name.lower()
        if "multirep" in low and "task-hopper" in low and "qslot" not in low:
            baseline = exp
            break
    if capped is None or baseline is None:
        print("  [info] q-slot comparison skipped (need both multirep task-hopper qslot2 & baseline).")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        {"exp": baseline, "label": "Q on all slots", "color": "#1b9e77", "ls": "-"},
        {"exp": capped, "label": "Q-slot cap = 2", "color": "#d95f02", "ls": "--"},
    ]
    save_figure(vc.plot_selection_rate_by_behavior(variants), out_dir / "selection_rate_by_behavior.png")
    save_figure(vc.plot_selection_rate_individual(variants), out_dir / "selection_rate_individual.png")
    save_figure(vc.plot_gir_development_by_behavior(variants), out_dir / "gir_development_by_behavior.png")
    save_figure(vc.plot_net_earnings_by_behavior(variants), out_dir / "net_earnings_by_behavior.png")
    save_figure(vc.plot_selection_fairness(variants), out_dir / "selection_fairness.png")
    save_figure(vc.plot_idle_streak(variants), out_dir / "idle_streak.png")
    save_figure(vc.plot_final_accuracy_per_task(variants), out_dir / "final_accuracy_per_task.png")
    for tt in TASK_TYPES:
        ds = DS_NAME[tt]
        save_figure(vc.plot_selection_rate_by_behavior(variants, tt),
                    out_dir / f"selection_rate_by_behavior_{ds}.png")
        save_figure(vc.plot_tr_development_by_behavior(variants, tt),
                    out_dir / f"tr_development_by_behavior_{ds}.png")
        save_figure(vc.plot_tr_development_individual(variants, tt),
                    out_dir / f"tr_development_individual_{ds}.png")
    print(f"  q-slot comparison: {baseline.name}  vs  {capped.name}")
    return 1


def main(root: Path, out: Path) -> None:
    global _OUT_BASE
    _OUT_BASE = out
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

    # Special Q-slot-cap comparison (task-hopper qslot2 vs uncapped baseline)
    total += _generate_qslot_graphs(experiments, out / "qslot-comparison")

    print(f"\nDone. {total} graphs written under {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate aggregate multirep comparison graphs")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Root folder of experiments (default: experiment/data/FinishedRuns)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder for png (default: figures/aggregate). "
                             "Non-png formats are written to sibling folders, e.g. figures/aggregate_svg.")
    parser.add_argument("--format", default="png",
                        help="Comma-separated output image format(s): png, svg, pdf "
                             "(default: png). Each figure is generated once and saved "
                             "to every requested format.")
    args = parser.parse_args()
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    for fmt in formats:
        if fmt not in ("png", "svg", "pdf"):
            parser.error(f"invalid --format '{fmt}' (choose from: png, svg, pdf)")
    set_figure_format(formats)
    if args.out is None:
        args.out = DEFAULT_OUT
    main(args.root, args.out)
