"""Generate all graphs for a multirep session.

Usage:
    python analysis/multirep_graphs.py <path-to-session.pkl> [--out <output-dir>]

Graphs are saved as PNG files under <output-dir> (defaults to a 'graphs/'
subfolder next to the session.pkl).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.multirep_loader import load_session
from analysis import multirep_plots as mrp
from analysis.plots import save_figure


def generate_all(session_pkl: Path, out_dir: Path | None = None) -> Path:
    session_pkl = Path(session_pkl)
    if session_pkl.is_dir():
        session_pkl = session_pkl / "session.pkl"
    session = load_session(session_pkl)
    rep = session.reputation_timeline

    if out_dir is None:
        out_dir = session_pkl.parent / "graphs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating graphs for '{session.preset_name}' ({session.n_tasks} tasks) → {out_dir}")

    if rep.empty:
        print("  [warn] reputation_timeline is empty — no per-task graphs.")
    else:
        # --- Per-user reputation evolution ---
        save_figure(mrp.plot_tr_over_tasks(rep),         out_dir / "tr_per_user.png")
        save_figure(mrp.plot_gir_over_tasks(rep),        out_dir / "gir_per_user.png")
        save_figure(mrp.plot_q_over_tasks(rep),          out_dir / "q_per_user.png")
        save_figure(mrp.plot_balance_over_tasks(rep),    out_dir / "balance_per_user.png")
        save_figure(mrp.plot_confidence_over_tasks(rep), out_dir / "confidence_per_user.png")

        # --- Group-level means ---
        save_figure(mrp.plot_tr_by_behavior(rep),        out_dir / "tr_by_behavior.png")
        save_figure(mrp.plot_gir_by_behavior(rep),       out_dir / "gir_by_behavior.png")
        save_figure(mrp.plot_q_by_behavior(rep),         out_dir / "q_by_behavior.png")
        save_figure(mrp.plot_balance_by_behavior(rep),   out_dir / "balance_by_behavior.png")

        # --- Selection analysis ---
        save_figure(mrp.plot_selection_frequency(rep),       out_dir / "selection_frequency.png")
        save_figure(mrp.plot_selection_heatmap(rep),         out_dir / "selection_heatmap.png")
        save_figure(mrp.plot_selection_score_over_tasks(rep),out_dir / "selection_score.png")
        save_figure(mrp.plot_score_vs_tr(rep),               out_dir / "score_vs_tr.png")

    # --- Per-task accuracy (only if embedded run_data exists) ---
    has_run_data = any(t.get("run_data") for t in session.tasks)
    if has_run_data:
        save_figure(mrp.plot_task_final_accuracy(session.tasks),  out_dir / "task_final_accuracy.png")
        save_figure(mrp.plot_task_accuracy_curves(session.tasks), out_dir / "task_accuracy_curves.png")
    else:
        print("  [info] No embedded run_data in session — skipping accuracy graphs.")

    files = sorted(out_dir.glob("*.png"))
    print(f"  Saved {len(files)} graphs to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multirep session graphs")
    parser.add_argument("session_pkl", type=Path, help="Path to session.pkl")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <session_dir>/graphs/)")
    args = parser.parse_args()
    generate_all(args.session_pkl, args.out)
