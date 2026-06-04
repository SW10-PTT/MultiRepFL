"""Generate all graphs for a multirep session.

Usage:
    python analysis/multirep_graphs.py <session.tar.gz>  [--out <output-dir>]
    python analysis/multirep_graphs.py <session-folder>  [--out <output-dir>]
    python analysis/multirep_graphs.py <session.pkl>     [--out <output-dir>]

Graphs are saved as PNG files under <output-dir>.  When not specified, the
default is a 'graphs/' subfolder inside the session folder (whether supplied
directly or inferred from the tarball name).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.multirep_loader import load_session, load_session_from_tarball
from analysis import multirep_plots as mrp
from analysis.plots import save_figure


def generate_all(source: Path, out_dir: Path | None = None) -> Path:
    source = Path(source)

    if source.suffixes[-2:] == [".tar", ".gz"]:
        session = load_session_from_tarball(source)
        # Default graphs/ folder sits next to the tarball, inside the session dir.
        default_out = source.parent / source.name.removesuffix(".tar.gz") / "graphs"
    else:
        session = load_session(source)  # handles folder or .pkl
        base = source if source.is_dir() else source.parent
        default_out = base / "graphs"

    if out_dir is None:
        out_dir = default_out
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = session.reputation_timeline
    print(f"Generating graphs for '{session.preset_name}' ({session.n_tasks} tasks) → {out_dir}")

    if rep.empty:
        print("  [warn] reputation_timeline is empty — no per-task graphs.")
    else:
        # --- TR split by task type (key divergence graph) ---
        save_figure(mrp.plot_tr_per_task_type(rep),             out_dir / "tr_per_task_type.png")
        save_figure(mrp.plot_tr_per_task_type_by_behavior(rep), out_dir / "tr_per_task_type_by_behavior.png")

        # --- Q split by task type ---
        save_figure(mrp.plot_q_per_task_type(rep),             out_dir / "q_per_task_type.png")
        save_figure(mrp.plot_q_per_task_type_by_behavior(rep), out_dir / "q_per_task_type_by_behavior.png")

        # --- Per-user reputation evolution ---
        save_figure(mrp.plot_tr_over_tasks(rep),         out_dir / "tr_per_user.png")
        save_figure(mrp.plot_gir_over_tasks(rep),        out_dir / "gir_per_user.png")
        save_figure(mrp.plot_balance_over_tasks(rep),    out_dir / "balance_per_user.png")
        save_figure(mrp.plot_confidence_over_tasks(rep), out_dir / "confidence_per_user.png")

        # --- Group-level means ---
        save_figure(mrp.plot_tr_by_behavior(rep),        out_dir / "tr_by_behavior.png")
        save_figure(mrp.plot_gir_by_behavior(rep),       out_dir / "gir_by_behavior.png")
        save_figure(mrp.plot_balance_by_behavior(rep),   out_dir / "balance_by_behavior.png")

        # --- Selection analysis ---
        save_figure(mrp.plot_selection_frequency(rep),        out_dir / "selection_frequency.png")
        save_figure(mrp.plot_selection_heatmap(rep),          out_dir / "selection_heatmap.png")
        save_figure(mrp.plot_selection_score_over_tasks(rep), out_dir / "selection_score.png")
        save_figure(mrp.plot_score_vs_tr(rep),                out_dir / "score_vs_tr.png")

    # --- Per-task accuracy ---
    if not session.global_accuracy.empty:
        save_figure(mrp.plot_accuracy_per_round_per_task(session.global_accuracy), out_dir / "task_accuracy_curves.png")
        save_figure(mrp.plot_final_accuracy_per_task(session.global_accuracy),     out_dir / "task_final_accuracy.png")
    else:
        # Fall back to extracting from embedded run_data (older session format)
        has_run_data = any(t.get("run_data") for t in session.tasks)
        if has_run_data:
            save_figure(mrp.plot_task_final_accuracy(session.tasks),  out_dir / "task_final_accuracy.png")
            save_figure(mrp.plot_task_accuracy_curves(session.tasks), out_dir / "task_accuracy_curves.png")
        else:
            print("  [info] No accuracy data in session — skipping accuracy graphs.")

    files = sorted(out_dir.glob("*.png"))
    print(f"  Saved {len(files)} graphs to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multirep session graphs")
    parser.add_argument(
        "source",
        type=Path,
        help="Session tarball (.tar.gz), session folder, or session.pkl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <session>/graphs/)",
    )
    args = parser.parse_args()
    generate_all(args.source, args.out)
