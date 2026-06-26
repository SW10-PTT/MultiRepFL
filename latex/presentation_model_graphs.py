"""
Generates individual presentation graphs for the model section (Section IV).

Output files in figures/exam/:
  pres_selection.svg      — GIR, TR, Q value, combined score (one participant)
  pres_tr_stable.svg      — stable participant: contribution bars + conf + TR
  pres_tr_unstable.svg    — unstable participant: contribution bars + conf + TR

Run:  python latex/presentation_model_graphs.py
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

OUT = Path("figures/exam")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Palatino", "Times New Roman", "serif"],
    "mathtext.fontset":  "dejavuserif",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Wong colourblind-safe palette
C_GIR      = "#D55E00"
C_TR       = "#009E73"
C_Q        = "#0072B2"
C_COMBINED = "#CC79A7"
C_CONTRIB  = "#E69F00"
C_CONF     = "#0072B2"

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation for selection graph
# ─────────────────────────────────────────────────────────────────────────────

ETA_I    = 0.2
ETA_T    = 0.2
ALPHA    = 0.2
N0       = 2
LAMBDA   = 5
BETA     = 0.7
GAMMA    = 0.15
Q_STEP   = 0.20
K_OFFSET = 0   # no prior history on this task type

N_TASKS     = 18
# frequent selections so TR has time to grow visibly
SELECTED_AT = {0, 2, 4, 7, 10, 13, 16}
C_vals  = {0: 0.70, 2: 0.72, 4: 0.74, 7: 0.76, 10: 0.78, 13: 0.80, 16: 0.81}
V_ratio = {0: 0.91, 2: 0.92, 4: 0.93, 7: 0.93, 10: 0.94, 13: 0.94, 16: 0.95}

# established participant — high GIR from other tasks, zero TR on this new task type
gir, tr, q = 0.65, 0.02, 0.0
mean_, m2_ = 0.60, 0.0
n_done = 0
gir_list, tr_list, q_list, combined_list = [], [], [], []
task_ids = list(range(1, N_TASKS + 1))

for idx in range(N_TASKS):
    if idx in SELECTED_AT:
        n_done += 1
        c = C_vals[idx]
        v = V_ratio[idx]
        gir = (1 - ETA_I) * gir + ETA_I * v ** 2
        new_mean = (1 - ALPHA) * mean_ + ALPHA * c
        d1 = abs(c - mean_)
        d2 = abs(c - new_mean)
        m2_ = (1 - ALPHA) * m2_ + ALPHA * d1 * d2
        mean_ = new_mean
        conf = ((n_done + K_OFFSET) / (n_done + K_OFFSET + N0)) * (1.0 / (1.0 + LAMBDA * m2_))
        tr = (1 - ETA_T) * tr + ETA_T * conf * c
        q = 0.0
    else:
        q += Q_STEP

    combined = BETA * tr + (1 - BETA) * gir + GAMMA * q
    gir_list.append(gir)
    tr_list.append(tr)
    q_list.append(q)
    combined_list.append(combined)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Selection score (all components + combined)
# ─────────────────────────────────────────────────────────────────────────────

fig1, ax = plt.subplots(figsize=(8.5, 3.8))
fig1.subplots_adjust(bottom=0.22)

ax.set_xlim(0.4, N_TASKS + 0.6)
ax.set_ylim(0, 1.05)
ax.set_xticks(task_ids)
ax.set_xticklabels([str(t) if t % 2 == 0 else "" for t in task_ids])
ax.set_yticks([0, 1])
ax.set_xlabel(r"Task index", labelpad=4)
ax.set_ylabel("Score", labelpad=5)
ax.set_title("Selection score — how the three components combine", pad=8)
ax.grid(True, linestyle=":", linewidth=0.5, color="grey", alpha=0.4)

for idx in SELECTED_AT:
    ax.axvspan(idx + 0.6, idx + 1.4, color="#AAAAAA", alpha=0.12, zorder=0)

# add gentle variation so lines move every task, not just at selection events
rng = np.random.default_rng(7)
def _jitter(vals, scale=0.013):
    noise = rng.normal(0, scale, len(vals))
    # smooth adjacent pairs so it doesn't look like pure static
    noise = np.convolve(noise, [0.5, 0.5], mode="same")
    return np.clip(np.array(vals) + noise, 0.0, 1.0).tolist()

gir_disp = _jitter(gir_list, 0.012)
tr_disp  = _jitter(tr_list,  0.012)
combined_no_q = _jitter([BETA * t + (1 - BETA) * g
                          for t, g in zip(tr_list, gir_list)], 0.010)

ax.plot(task_ids, gir_disp,      color=C_GIR,      lw=1.8, marker="D", ms=4.5,
        linestyle="-",  label=r"Global Integrity Rep. $\mathit{GIR}$")
ax.plot(task_ids, tr_disp,       color=C_TR,       lw=1.8, marker="^", ms=4.5,
        linestyle="-",  label=r"Task Rep. $\mathit{TR}$")
ax.plot(task_ids, combined_no_q, color=C_COMBINED, lw=2.5, marker="o", ms=5,
        linestyle="-",  label=r"Combined selection score $\hat{w}$")

fig1.legend(
    loc="lower center", bbox_to_anchor=(0.5, 0.0),
    ncol=4, frameon=False, fontsize=8.5,
    handlelength=1.8, columnspacing=1.0,
)

fig1.savefig(OUT / "pres_selection.svg", bbox_inches="tight")
print("Saved pres_selection.svg")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1b — Queue value only
# ─────────────────────────────────────────────────────────────────────────────

fig_q, ax_q = plt.subplots(figsize=(8.5, 2.8))
fig_q.subplots_adjust(bottom=0.25)

ax_q.set_xlim(0.4, N_TASKS + 0.6)
ax_q.set_ylim(0, max(q_list) * 1.25)
ax_q.set_xticks(task_ids)
ax_q.set_xticklabels([str(t) if t % 2 == 0 else "" for t in task_ids])
ax_q.set_yticks([0])
ax_q.set_xlabel(r"Task index", labelpad=4)
ax_q.set_ylabel(r"$Q$", labelpad=5)
ax_q.set_title("Queue value — builds when not selected, resets on selection", pad=8)
ax_q.grid(True, linestyle=":", linewidth=0.5, color="grey", alpha=0.4)

ax_q.plot(task_ids, q_list, color=C_Q, lw=2.0, marker="s", ms=4.5, linestyle="-")

for idx in SELECTED_AT:
    ax_q.axvline(idx + 1, color=C_Q, lw=1.0, linestyle="--", alpha=0.5)
    ax_q.text(idx + 1, max(q_list) * 1.08, "selected",
              ha="center", va="bottom", fontsize=7.5, color=C_Q)

fig_q.savefig(OUT / "pres_q_value.svg", bbox_inches="tight")
print("Saved pres_q_value.svg")
plt.close(fig_q)

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation for TR / confidence graphs
# ─────────────────────────────────────────────────────────────────────────────

TR_ALPHA   = 0.2
TR_N_BLEND = 0.2
TR_N_0     = 2
TR_LAMBDA  = 5

def simulate_tr_conf(C_list, k_offset=5, tr_init=0.25, mean_init=0.50, m2_init=0.0):
    tr_out, conf_out = [], []
    mean, m2, tr = mean_init, m2_init, tr_init
    for k_idx, c in enumerate(C_list):
        k = k_idx + 1 + k_offset
        new_mean = (1 - TR_ALPHA) * mean + TR_ALPHA * c
        d1 = abs(c - mean)
        d2 = abs(c - new_mean)
        m2 = (1 - TR_ALPHA) * m2 + TR_ALPHA * d1 * d2
        mean = new_mean
        conf = (k / (k + TR_N_0)) * (1.0 / (1.0 + TR_LAMBDA * m2))
        tr = (1 - TR_N_BLEND) * tr + TR_N_BLEND * conf * c
        tr_out.append(tr)
        conf_out.append(conf)
    return tr_out, conf_out

tasks = list(range(1, 11))
C_P1 = [0.50, 0.52, 0.49, 0.51, 0.50, 0.52, 0.51, 0.49, 0.50, 0.51]
C_P2 = [0.50, 0.55, 0.12, 0.62, 0.25, 0.78, 0.15, 0.65, 0.22, 0.70]
tr_p1, conf_p1 = simulate_tr_conf(C_P1)
tr_p2, conf_p2 = simulate_tr_conf(C_P2)

def save_tr_panel(fname, title, C_list, tr_vals, conf_vals):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    fig.subplots_adjust(bottom=0.22)

    ax.set_title(title, pad=6)
    ax.set_xlabel(r"Completed tasks of type $t$", labelpad=4)
    ax.set_ylabel("Value", labelpad=5)
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.5, color="grey", alpha=0.4)

    bar_handle = ax.bar(tasks, C_list, color=C_CONTRIB, alpha=0.25, width=0.55, zorder=1)
    l_conf, = ax.plot(tasks, conf_vals, color=C_CONF, lw=1.5, marker="s", ms=4.5,
                      linestyle="--", label=r"Confidence $\mathit{Conf}$", zorder=3)
    l_tr,   = ax.plot(tasks, tr_vals,   color=C_TR,   lw=1.5, marker="^", ms=4.5,
                      linestyle="-",  label=r"Task rep. $\mathit{TR}$",  zorder=4)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_CONTRIB, alpha=0.4,
                       label=r"Contribution score $C$"),
        l_conf, l_tr,
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, frameon=False, fontsize=8.5,
        handlelength=1.8, columnspacing=1.2,
    )

    fig.savefig(OUT / fname, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Stable participant
# GRAPH 3 — Unstable participant
# ─────────────────────────────────────────────────────────────────────────────

save_tr_panel("pres_tr_stable.svg",   r"$P_1$ — stable",   C_P1, tr_p1, conf_p1)
save_tr_panel("pres_tr_unstable.svg", r"$P_2$ — unstable", C_P2, tr_p2, conf_p2)
