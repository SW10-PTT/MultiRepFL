#!/usr/bin/env python3
"""Report per-PC experiment run times, split by dataset (mnist/cifar).

Walks analysis/logs/**/*.tar.gz. For each run extracts:
  - PC name from the filename (node_id = hostname[:6], normalized so that
    truncation artifacts like "anton"/"anton-" or "Pangea"/"Pangea-2" group
    under the same PC)
  - dataset (mnist/cifar) from the archive's top-level folder name
  - "TOTAL EXPERIMENT TIME: X seconds" comment from the run's csv

Reports, per PC and dataset, the median task time (used instead of the mean
since it is robust to occasional very slow/fast outlier runs), plus a
separate list of detected outliers (IQR method, only for PC/dataset groups
with >= 4 samples).
"""

import re
import statistics
import tarfile
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"

TIME_RE = re.compile(r"TOTAL EXPERIMENT TIME:\s*([\d.]+)\s*seconds")

OUTLIER_MIN_SAMPLES = 4
OUTLIER_IQR_MULT = 1.5


def normalize_pc_name(node_id: str) -> str:
    name = node_id.rstrip("-")
    name = re.sub(r"-?\d+$", "", name)
    return name.rstrip("-")


def parse_pc_name(tar_path: Path) -> str:
    stem = tar_path.name[: -len(".tar.gz")]
    parts = stem.split("-")
    node_id = "-".join(parts[3:-1])
    return normalize_pc_name(node_id)


def load_records():
    records = []  # (pc, dataset, time_seconds, path)
    for f in sorted(LOGS_DIR.rglob("*.tar.gz")):
        pc = parse_pc_name(f)
        try:
            with tarfile.open(f, "r:gz") as tf:
                names = tf.getnames()
                top = names[0].split("/")[0]
                if top.startswith("cifar"):
                    dataset = "cifar"
                elif top.startswith("mnist"):
                    dataset = "mnist"
                else:
                    dataset = top.split("-")[0]
                csv_name = next(n for n in names if n.endswith(".csv"))
                data = tf.extractfile(csv_name).read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue

        match = None
        for line in reversed(data.splitlines()):
            match = TIME_RE.search(line)
            if match:
                break
        if not match:
            print(f"  [skip] {f}: no TOTAL EXPERIMENT TIME found")
            continue

        records.append((pc, dataset, float(match.group(1)), f))
    return records


def find_outliers(times_with_paths):
    """Return entries whose time falls outside [Q1 - k*IQR, Q3 + k*IQR]."""
    if len(times_with_paths) < OUTLIER_MIN_SAMPLES:
        return []
    times = sorted(t for t, _ in times_with_paths)
    q1, q3 = statistics.quantiles(times, n=4)[0], statistics.quantiles(times, n=4)[2]
    iqr = q3 - q1
    if iqr == 0:
        return []
    lo, hi = q1 - OUTLIER_IQR_MULT * iqr, q3 + OUTLIER_IQR_MULT * iqr
    return [(t, p) for t, p in times_with_paths if t < lo or t > hi]


def main():
    records = load_records()
    if not records:
        print("No runs found under", LOGS_DIR)
        return

    grand_total = sum(r[2] for r in records)

    # group[(pc, dataset)] -> list of (time, path)
    groups = {}
    for pc, dataset, t, path in records:
        groups.setdefault((pc, dataset), []).append((t, path))

    pcs = sorted({pc for pc, _, _, _ in records})

    # per-pc stats for sorting + display
    pc_stats = {}
    for pc in pcs:
        per_dataset = {}
        all_times = []
        for dataset in ("cifar", "mnist"):
            entries = groups.get((pc, dataset), [])
            if entries:
                times = [t for t, _ in entries]
                times_sorted = sorted(times)
                n_top1 = max(1, round(len(times) * 0.01))
                per_dataset[dataset] = {
                    "n": len(times),
                    "median": statistics.median(times),
                    "mean": statistics.mean(times),
                    "top1pct_n": n_top1,
                    "top1pct_mean": statistics.mean(times_sorted[:n_top1]),
                }
                all_times.extend(times)
        if "cifar" in per_dataset:
            sort_key = per_dataset["cifar"]["median"]
        elif "mnist" in per_dataset:
            sort_key = per_dataset["mnist"]["median"]
        else:
            sort_key = float("inf")
        pc_stats[pc] = {
            "per_dataset": per_dataset,
            "total_time": sum(all_times),
            "n_total": len(all_times),
            "sort_key": sort_key,
        }

    pcs_sorted = sorted(pcs, key=lambda pc: pc_stats[pc]["sort_key"])

    print("=" * 80)
    print("PC SPEED REPORT")
    print("=" * 80)
    print(f"Total runs: {len(records)}")
    print(f"Total experiment time (all PCs, all runs): {grand_total:.2f} s "
          f"({grand_total / 3600:.2f} h)")
    print()
    print("PCs sorted fastest to slowest (by cifar median task time)")
    print("-" * 80)
    header = f"{'PC':<10} {'Cifar median (n)':<20} {'Mnist median (n)':<20} {'Total time (s)':<15}"
    print(header)
    print("-" * 80)
    for pc in pcs_sorted:
        stats = pc_stats[pc]
        cifar = stats["per_dataset"].get("cifar")
        mnist = stats["per_dataset"].get("mnist")
        cifar_str = f"{cifar['median']:.1f}s (n={cifar['n']})" if cifar else "-"
        mnist_str = f"{mnist['median']:.1f}s (n={mnist['n']})" if mnist else "-"
        print(f"{pc:<10} {cifar_str:<20} {mnist_str:<20} {stats['total_time']:.1f}")

    print()
    print("Average (mean) times per PC, for reference")
    print("-" * 80)
    print(header)
    print("-" * 80)
    for pc in pcs_sorted:
        stats = pc_stats[pc]
        cifar = stats["per_dataset"].get("cifar")
        mnist = stats["per_dataset"].get("mnist")
        cifar_str = f"{cifar['mean']:.1f}s (n={cifar['n']})" if cifar else "-"
        mnist_str = f"{mnist['mean']:.1f}s (n={mnist['n']})" if mnist else "-"
        print(f"{pc:<10} {cifar_str:<20} {mnist_str:<20} {stats['total_time']:.1f}")

    print()
    print("Best-case times per PC: mean of each PC's top 1% fastest runs, per dataset")
    print("(sorted by cifar top-1% time)")
    print("-" * 80)
    print(header)
    print("-" * 80)
    pcs_by_top1pct = sorted(
        pcs,
        key=lambda pc: pc_stats[pc]["per_dataset"].get("cifar", {}).get("top1pct_mean", float("inf")),
    )
    for pc in pcs_by_top1pct:
        stats = pc_stats[pc]
        cifar = stats["per_dataset"].get("cifar")
        mnist = stats["per_dataset"].get("mnist")
        cifar_str = f"{cifar['top1pct_mean']:.1f}s (n={cifar['top1pct_n']})" if cifar else "-"
        mnist_str = f"{mnist['top1pct_mean']:.1f}s (n={mnist['top1pct_n']})" if mnist else "-"
        print(f"{pc:<10} {cifar_str:<20} {mnist_str:<20} {stats['total_time']:.1f}")

    print()
    print("=" * 80)
    print(f"OUTLIERS (IQR method, {OUTLIER_IQR_MULT}x IQR, groups with "
          f">= {OUTLIER_MIN_SAMPLES} runs)")
    print("=" * 80)
    any_outliers = False
    for (pc, dataset), entries in sorted(groups.items()):
        outliers = find_outliers(entries)
        for t, path in sorted(outliers):
            any_outliers = True
            print(f"{pc:<10} {dataset:<6} {t:>10.2f}s  {path}")
    if not any_outliers:
        print("(none)")

    print()
    print("=" * 80)
    print("TOP 1% FASTEST RUNS (per dataset)")
    print("=" * 80)
    for dataset in ("cifar", "mnist"):
        entries = [(t, pc, path) for (pc, ds), grp in groups.items() if ds == dataset
                   for t, path in grp]
        entries.sort()
        n_top = max(1, len(entries) // 100)
        print(f"-- {dataset} (top {n_top} of {len(entries)}) --")
        for t, pc, path in entries[:n_top]:
            print(f"{pc:<10} {t:>10.2f}s  {path}")


if __name__ == "__main__":
    main()
