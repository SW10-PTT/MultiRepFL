import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Palatino", "Times New Roman", "serif"],
    "mathtext.fontset":  "dejavuserif",
    "font.size":         9.5,
    "axes.titlesize":    9.5,
    "axes.labelsize":    9.5,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# --- Contract constants (from OpenFLChallenge.sol) ---
TR_ALPHA   = 0.2
TR_N_BLEND = 0.2
TR_N_0     = 2   # matches experiment config (Table 4)
TR_LAMBDA  = 5   # matches experiment config (Table 4)
TR_GIR_LR  = 0.2

def simulate(C_list, V_ratio_list, tr_init=0.0, gir_init=0.0, mean_init=0.0, m2_init=0.0, k_offset=0):
    conf_list, tr_list, gir_list = [], [], []
    mean, m2, tr, gir = mean_init, m2_init, tr_init, gir_init
    for k_idx, (c, v) in enumerate(zip(C_list, V_ratio_list)):
        k  = k_idx + 1 + k_offset
        d1 = abs(c - mean)
        new_mean = c if k == 1 else (1 - TR_ALPHA) * mean + TR_ALPHA * c
        d2   = abs(c - new_mean)
        new_m2 = (1 - TR_ALPHA) * m2 + TR_ALPHA * d1 * d2
        mean, m2 = new_mean, new_m2
        conf = (k / (k + TR_N_0)) * (1.0 / (1.0 + TR_LAMBDA * m2))
        conf_list.append(conf)
        tr = (1 - TR_N_BLEND) * tr + TR_N_BLEND * conf * c
        tr_list.append(tr)
        gir = (1 - TR_GIR_LR) * gir + TR_GIR_LR * v ** 2
        gir_list.append(gir)
    return conf_list, tr_list, gir_list

tasks = list(range(1, 11))

C_P1       = [0.50, 0.52, 0.49, 0.51, 0.50, 0.52, 0.51, 0.49, 0.50, 0.51]
V_ratio_P1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

C_P2       = [0.50, 0.55, 0.12, 0.62, 0.25, 0.78, 0.15, 0.65, 0.22, 0.70]
V_ratio_P2 = [1, 1, 0.25, 1, 0.5, 1, 0.25, 1, 0.5, 1]

INIT = dict(tr_init=0.25, gir_init=0.60, mean_init=0.50, m2_init=0.0)
K_OFFSET = 5  # represents ~5 prior tasks already completed

conf_p1, tr_p1, gir_p1 = simulate(C_P1, V_ratio_P1, **INIT, k_offset=K_OFFSET)
conf_p2, tr_p2, gir_p2 = simulate(C_P2, V_ratio_P2, **INIT, k_offset=K_OFFSET)

# Wong (2011) colorblind-safe — no purple
metrics = [
    {"label": r"Contribution score $C$",     "color": "#E69F00", "marker": "o", "P1": C_P1,    "P2": C_P2    },
    {"label": r"Confidence $\mathit{Conf}$",  "color": "#0072B2", "marker": "s", "P1": conf_p1, "P2": conf_p2 },
    {"label": r"Task rep. $\mathit{TR}$",     "color": "#009E73", "marker": "^", "P1": tr_p1,   "P2": tr_p2   },
    {"label": r"Integrity rep. $\mathit{GIR}$","color": "#D55E00", "marker": "D", "P1": gir_p1,  "P2": gir_p2  },
]

out_dir = Path("figures/exam")
out_dir.mkdir(parents=True, exist_ok=True)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
fig.subplots_adjust(wspace=0.08, bottom=0.22)

panels = [(ax_l, "P1", r"$P_1$ — stable"), (ax_r, "P2", r"$P_2$ — unstable")]

legend_handles = []

for ax, participant, title in panels:
    ax.set_title(title, pad=6)
    ax.set_xlabel(r"Completed tasks of type $t$", labelpad=4)
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.5, color="grey", alpha=0.4)

    for m in metrics:
        line, = ax.plot(
            tasks, m[participant],
            linestyle="-",
            marker=m["marker"],
            linewidth=1.5,
            markersize=4.5,
            color=m["color"],
            label=m["label"],
        )
        if ax is ax_l:
            legend_handles.append(line)

ax_l.set_ylabel("Value", labelpad=5)

# Single shared legend in one row below both panels
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    frameon=False,
    fontsize=8.5,
    handlelength=1.8,
    columnspacing=1.2,
)

fig.savefig(out_dir / "example_summary.svg", bbox_inches="tight")
print("Saved example_summary.svg")
plt.close(fig)

# --- Split panels ---
for participant, title, fname in [
    ("P1", r"$P_1$ — stable",   "example_stable.svg"),
    ("P2", r"$P_2$ — unstable", "example_unstable.svg"),
]:
    fig_s, ax_s = plt.subplots(figsize=(5.0, 3.4))
    fig_s.subplots_adjust(bottom=0.38)

    ax_s.set_title(title, pad=6)
    ax_s.set_xlabel(r"Completed tasks of type $t$", labelpad=4)
    ax_s.set_ylabel("Value", labelpad=5)
    ax_s.set_xticks(tasks)
    ax_s.set_xlim(0.5, 10.5)
    ax_s.set_ylim(0, 1.05)
    ax_s.grid(True, linestyle=":", linewidth=0.5, color="grey", alpha=0.4)

    handles = []
    for m in metrics:
        line, = ax_s.plot(
            tasks, m[participant],
            linestyle="-", marker=m["marker"],
            linewidth=1.5, markersize=4.5,
            color=m["color"], label=m["label"],
        )
        handles.append(line)

    fig_s.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=2, frameon=False, fontsize=8.5,
        handlelength=1.8, columnspacing=1.2,
    )

    fig_s.savefig(out_dir / fname, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig_s)
