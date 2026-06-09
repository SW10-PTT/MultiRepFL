"""
Run multiple multirep preset files sequentially, in parallel, or in tmux windows.

Usage:
    python experiment/batch_runner.py preset1.json preset2.json [options]

    --parallel      Run all presets concurrently (each gets its own blockchain node).
    --tmux          Launch each preset in its own tmux window (session: multirep-batch).
    --anvil         Start an Anvil node for each run.
    --ganache       Start a Ganache node for each run.
    --graphs        Generate graphs after each run completes.
    --seed N        Override the RNG seed for every preset in this batch (fresh
                    seed per batch = independent runs for honest variance bands).

Examples:
    # Sequential (default)
    python experiment/batch_runner.py \\
        experiment/presets/EXP-multirep-mixed-distribution-5-task-dataset-switch.json \\
        experiment/presets/EXP-globalrep-mixed-distribution-5-task-dataset-switch.json \\
        --anvil

    # Parallel
    python experiment/batch_runner.py \\
        experiment/presets/EXP-multirep-mixed-distribution-5-task-dataset-switch.json \\
        experiment/presets/EXP-globalrep-mixed-distribution-5-task-dataset-switch.json \\
        --parallel --anvil

    # Tmux (each run in its own inspectable window)
    python experiment/batch_runner.py \\
        experiment/presets/EXP-multirep-mixed-distribution-5-task-dataset-switch.json \\
        experiment/presets/EXP-globalrep-mixed-distribution-5-task-dataset-switch.json \\
        --tmux --anvil
"""

import argparse
import shlex
import subprocess
import sys
import threading
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = str(_REPO_ROOT / ".venv" / "bin" / "python")


def _build_cmd(preset_file: str, anvil: bool, ganache: bool, graphs: bool,
               seed: int | None = None) -> list[str]:
    script = str(_REPO_ROOT / "experiment" / "multirep.py")
    cmd = [_PYTHON, script, "--preset", preset_file]
    if anvil:
        cmd.append("--anvil")
    elif ganache:
        cmd.append("--ganache")
    if graphs:
        cmd.append("--graphs")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def _stream_output(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="", flush=True)


def run_sequential(
    presets: list[str],
    anvil: bool,
    ganache: bool,
    graphs: bool,
    seed: int | None = None,
) -> dict[str, int]:
    results: dict[str, int] = {}
    for preset in presets:
        name = Path(preset).stem
        print(f"\n{'='*60}")
        print(f"Starting: {name}")
        print(f"{'='*60}\n")
        cmd = _build_cmd(preset, anvil, ganache, graphs, seed)
        proc = subprocess.run(cmd, cwd=_REPO_ROOT)
        results[name] = proc.returncode
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"\n[{name}] {status}")
    return results


_LAUNCH_STAGGER_S = 5  # seconds between subprocess launches to avoid port-binding races


def run_parallel(
    presets: list[str],
    anvil: bool,
    ganache: bool,
    graphs: bool,
    seed: int | None = None,
) -> dict[str, int]:
    import time

    procs: list[tuple[str, subprocess.Popen]] = []
    threads: list[threading.Thread] = []

    needs_stagger = anvil or ganache

    try:
        for i, preset in enumerate(presets):
            name = Path(preset).stem
            cmd = _build_cmd(preset, anvil, ganache, graphs, seed)
            print(f"Launching: {name}")
            proc = subprocess.Popen(
                cmd,
                cwd=_REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            procs.append((name, proc))
            t = threading.Thread(target=_stream_output, args=(proc, name), daemon=False)
            t.start()
            threads.append(t)
            if needs_stagger and i < len(presets) - 1:
                time.sleep(_LAUNCH_STAGGER_S)

        print(f"\nAll {len(procs)} runs launched. Waiting for completion...\n")

        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("\nInterrupted — terminating all runs...", file=sys.stderr)
        for _, proc in procs:
            proc.terminate()
        for _, proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for t in threads:
            t.join(timeout=2)
        sys.exit(1)

    results: dict[str, int] = {}
    for name, proc in procs:
        proc.wait()
        results[name] = proc.returncode

    return results


_TMUX_SESSION = "multirep-batch"


def _next_tmux_session(base: str) -> str:
    """Return base if unused, else base-1, base-2, … until a free name is found."""
    existing = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True, text=True,
    ).stdout.split()
    if base not in existing:
        return base
    i = 1
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"


def run_tmux(presets: list[str], anvil: bool, ganache: bool, graphs: bool,
             seed: int | None = None) -> None:
    import shutil
    import time

    if not shutil.which("tmux"):
        print("ERROR: tmux not found in PATH", file=sys.stderr)
        sys.exit(1)

    session = _next_tmux_session(_TMUX_SESSION)

    for i, preset in enumerate(presets):
        name = Path(preset).stem
        cmd = _build_cmd(preset, anvil, ganache, graphs, seed)
        shell_cmd = " ".join(shlex.quote(c) for c in cmd)
        shell_script = f"{shell_cmd}; echo; echo '=== DONE (exit $?) ==='; exec bash"

        if i == 0:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, "-n", name,
                 "--", "bash", "-c", shell_script],
                check=True,
            )
        else:
            if anvil or ganache:
                time.sleep(_LAUNCH_STAGGER_S)
            subprocess.run(
                ["tmux", "new-window", "-t", f"{session}:", "-n", name,
                 "--", "bash", "-c", shell_script],
                check=True,
            )

    print(f"\nLaunched {len(presets)} runs in tmux session '{session}'.")
    print(f"\n  tmux attach -t {session}\n")
    print("Navigate windows: Ctrl-b n (next)  Ctrl-b p (prev)  Ctrl-b w (list)")


def _print_summary(results: dict[str, int]) -> None:
    print(f"\n{'='*60}")
    print("Batch summary")
    print(f"{'='*60}")
    for name, code in results.items():
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"  {name}: {status}")
    failed = [n for n, c in results.items() if c != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} runs succeeded.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple multirep presets sequentially or in parallel."
    )
    parser.add_argument(
        "presets",
        nargs="+",
        help="Paths to preset JSON files.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all presets concurrently.",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Launch each preset in its own tmux window (session: multirep-batch).",
    )
    blockchain_group = parser.add_mutually_exclusive_group()
    blockchain_group.add_argument(
        "--anvil",
        action="store_true",
        help="Start an Anvil node for each run.",
    )
    blockchain_group.add_argument(
        "--ganache",
        action="store_true",
        help="Start a Ganache node for each run.",
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Generate graphs after each run completes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the RNG seed for every preset in this batch (passed to each "
             "multirep run as --seed). Use a fresh seed per batch for independent runs.",
    )
    args = parser.parse_args()

    missing = [p for p in args.presets if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"ERROR: preset not found: {p}", file=sys.stderr)
        sys.exit(1)

    if args.tmux:
        run_tmux(args.presets, args.anvil, args.ganache, args.graphs, args.seed)
        return

    if args.parallel:
        results = run_parallel(args.presets, args.anvil, args.ganache, args.graphs, args.seed)
    else:
        results = run_sequential(args.presets, args.anvil, args.ganache, args.graphs, args.seed)

    _print_summary(results)

    if any(c != 0 for c in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
