"""Throwaway: mixed-distribution verification — selection rates, merit, TR,
earnings by specialist group, straight from FinishedRuns."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.multirep_aggregate_loader import (
    CIFAR_TT, MNIST_TT, discover_experiments,
)

ROOT = Path(__file__).resolve().parent.parent / "experiment" / "data" / "FinishedRuns"
OUT = Path(__file__).resolve().parent.parent / "figures" / "tab_mixed_verification.png"

SYS_LABELS = {"globalrep": "GlobalRep", "multirep": "MultiRep"}
DS_COLORS = {"MNIST": "#1f77b4", "CIFAR-10": "#ff7f0e"}
GROUPS = ["MNIST-strong", "Average", "CIFAR-strong"]

exps = {e.system: e for e in discover_experiments(ROOT) if "mixed" in e.name}


def group_of(name: str) -> str:
    for g in GROUPS:
        if g.split("-")[0] in name:
            return g
    return "?"


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, system in zip(axes, ["globalrep", "multirep"]):
    rep = exps[system].reputation_timeline()
    rep = rep.assign(grp=rep["user_name"].map(group_of))
    x = np.arange(len(GROUPS))
    w = 0.36
    for j, (tt, ds) in enumerate([(MNIST_TT, "MNIST"), (CIFAR_TT, "CIFAR-10")]):
        vals, labels = [], []
        for g in GROUPS:
            s = rep[(rep["grp"] == g) & (rep["task_type"] == tt)]
            k, n = int(s["was_selected"].sum()), len(s)
            vals.append(k / n)
            labels.append(f"{k}/{n}")
        star = [0] if ds == "MNIST" else [2]  # data-rich group index
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=DS_COLORS[ds],
                      edgecolor="black", linewidth=0.6, alpha=0.9, label=ds)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008,
                    f"{v*100:.1f}%", ha="center", fontsize=8)
        for i in star:
            ax.annotate("★", (x[i] + (j - 0.5) * w, 0), ha="center", va="bottom",
                        color="#444", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_title(f"{SYS_LABELS[system]}  ({exps[system].n_runs} runs)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.62)
axes[0].set_ylabel("Selection rate on that dataset's tasks")
axes[1].legend(title="Task dataset", fontsize=9)
fig.suptitle("Mixed distribution — realized selection rate by specialist group "
             "(★ = group's data-rich dataset)")
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=150)
    print("saved", OUT.with_suffix(f".{ext}"))

# ---- exact numbers for the paper table ----
for system in ["globalrep", "multirep"]:
    e = exps[system]
    rep = e.reputation_timeline().assign(grp=lambda d: d["user_name"].map(group_of))
    print("=" * 20, system)
    for g in GROUPS:
        row = []
        for tt in (MNIST_TT, CIFAR_TT):
            s = rep[(rep["grp"] == g) & (rep["task_type"] == tt)]
            row.append(f"sel {s['was_selected'].mean()*100:.1f}% "
                       f"merit {s['selection_score'].mean():.3f}")
        print(f"  {g:13s} MNIST: {row[0]}   CIFAR: {row[1]}")
    # final TR + earnings by group
    last = rep.sort_values("task_index").groupby(["run", "user_name"]).last().reset_index()
    last["grp"] = last["user_name"].map(group_of)
    if system == "multirep":
        last["trM"] = last["tr_all_post"].apply(lambda d: d.get(MNIST_TT, 0.0))
        last["trC"] = last["tr_all_post"].apply(lambda d: d.get(CIFAR_TT, 0.0))
    else:
        last["trM"] = last["tr_post"]; last["trC"] = last["tr_post"]
    agg = last.groupby("grp").agg(trM=("trM", "mean"), trC=("trC", "mean"),
                                  eth=("balance_post", "mean")).round(3)
    print(agg.to_string())
    # rho per-user mean
    ftr = last.groupby("user_name")[["trM", "trC"]].mean()
    print("  rho =", round(ftr["trM"].rank().corr(ftr["trC"].rank()), 2))
