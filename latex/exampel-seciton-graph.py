import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

# Match LaTeX document typography as closely as possible without requiring a
# full TeX installation.
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Palatino", "Times New Roman", "serif"],
    "mathtext.fontset":  "dejavuserif",
    "font.size":         12,
    "axes.titlesize":    12,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Data
tasks = [1, 2, 3, 4, 5]

# Wong (2011) colorblind-safe palette
_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

data = {
    "C": {
        "P1": [0.80, 0.82, 0.79, 0.81, 0.80],
        "P2": [0.80, 0.84, 0.10, 0.90, 0.85],
        "label": r"Contribution score $C$",
        "marker": "o",
    },
    "Conf": {
        "P1": [0.167, 0.285, 0.374, 0.444, 0.499],
        "P2": [0.167, 0.284, 0.144, 0.181, 0.224],
        "label": r"Confidence $\mathit{Conf}$",
        "marker": "s",
    },
    "TaskRep": {
        "P1": [0.027, 0.068, 0.114, 0.163, 0.210],
        "P2": [0.027, 0.069, 0.058, 0.079, 0.101],
        "label": r"Task reputation $\mathit{TR}$",
        "marker": "^",
    },
    "I": {
        "P1": [0.200, 0.360, 0.488, 0.590, 0.672],
        "P2": [0.200, 0.360, 0.338, 0.470, 0.576],
        "label": r"Integrity reputation $\mathit{GIR}$",
        "marker": "D",
    },
}

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6.5, 4.6))

metric_handles = []

for (metric, values), color in zip(data.items(), _COLORS):
    line_p1, = ax.plot(
        tasks, values["P1"],
        linestyle="-",
        marker=values["marker"],
        linewidth=1.6,
        markersize=5,
        color=color,
        label=values["label"],
    )
    ax.plot(
        tasks, values["P2"],
        linestyle="--",
        marker=values["marker"],
        linewidth=1.6,
        markersize=5,
        color=color,
    )
    metric_handles.append(line_p1)

ax.set_xlabel("Completed tasks", labelpad=6)
ax.set_ylabel("Value", labelpad=6)
ax.set_xticks(tasks)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle=":", linewidth=0.5, color="grey", alpha=0.4)


# Legend 1 — metric lines
legend_metrics = ax.legend(
    handles=metric_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
    frameon=False,
    fontsize=11,
    handlelength=2.0,
)
ax.add_artist(legend_metrics)

# Legend 2 — participant line style
participant_handles = [
    Line2D([0], [0], linestyle="-",  linewidth=1.6, color="black", label=r"$P_1$ (stable)"),
    Line2D([0], [0], linestyle="--", linewidth=1.6, color="black", label=r"$P_2$ (unstable)"),
]
ax.legend(
    handles=participant_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.26),
    ncol=2,
    frameon=False,
    fontsize=11,
    handlelength=2.0,
)

fig.tight_layout()

fig.savefig(out_dir / "example_summary.svg", bbox_inches="tight")

plt.show()