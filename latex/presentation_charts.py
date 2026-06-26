"""
Generates two presentation-quality chart PNGs for the 4-minute model section.

Output files (in figures/):
  pres_flow.png       — flow diagram: GIR + TR + Q Value → Selection
  pres_tr_compare.png — TR comparison: stable vs unstable participant

Run: python latex/presentation_charts.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "font.size": 14,
})

# colours (Wong colourblind-safe)
C_GIR   = "#D55E00"   # burnt orange
C_TR    = "#009E73"   # teal
C_Q     = "#0072B2"   # blue
C_SEL   = "#CC79A7"   # pink
C_ARROW = "#555555"

# ── 1. FLOW DIAGRAM ───────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, subtext, colour, fontsize=13):
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04", linewidth=1.8,
        edgecolor=colour, facecolor=colour + "22",
    )
    ax.add_patch(box)
    ax.text(x, y + 0.04, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=colour)
    if subtext:
        ax.text(x, y - 0.15, subtext,
                ha="center", va="center", fontsize=10,
                color="#444444", style="italic")

def draw_arrow(ax, x0, y0, x1, y1, colour=C_ARROW):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=colour,
            lw=1.8, mutation_scale=16,
        ),
    )

fig1, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3.5)
ax1.axis("off")

# ── input boxes (left column)
BOX_W, BOX_H = 2.4, 0.88
Y_GIR = 2.8
Y_TR  = 1.75
Y_Q   = 0.7

draw_box(ax1, 2.0, Y_GIR, BOX_W, BOX_H,
         "Global Integrity", "honesty across ALL tasks", C_GIR)
draw_box(ax1, 2.0, Y_TR,  BOX_W, BOX_H,
         "Task Reputation", "performance on THIS task type", C_TR)
# small confidence note inside TR box, below the subtext
ax1.text(2.0, Y_TR - 0.36, "confidence-weighted",
         ha="center", va="center", fontsize=8.5,
         color=C_TR, style="italic",
         bbox=dict(boxstyle="round,pad=0.15", fc="#EAFAF4", ec=C_TR, lw=0.8))
draw_box(ax1, 2.0, Y_Q,   BOX_W, BOX_H,
         "Queue Value", "time since last selected", C_Q)

# ── merge node
MX, MY = 5.5, 1.75
merge_circ = plt.Circle((MX, MY), 0.38, linewidth=1.8,
                         edgecolor=C_SEL, facecolor=C_SEL + "22", zorder=3)
ax1.add_patch(merge_circ)
ax1.text(MX, MY, "+", ha="center", va="center",
         fontsize=22, color=C_SEL, fontweight="bold", zorder=4)

# arrows → merge
for ysrc in [Y_GIR, Y_TR, Y_Q]:
    draw_arrow(ax1, 2.0 + BOX_W / 2, ysrc, MX - 0.38, MY, C_ARROW)

# ── selection score box
draw_box(ax1, 7.9, MY, 2.6, BOX_H,
         "Selection Score", "top-N participants chosen", C_SEL, fontsize=13)
draw_arrow(ax1, MX + 0.38, MY, 7.9 - 1.3, MY)


# title
ax1.set_title("How participant selection works",
              fontsize=16, fontweight="bold", pad=10, color="#222222")

fig1.tight_layout()
fig1.savefig(OUT / "pres_flow.png", dpi=180, bbox_inches="tight")
print("Saved pres_flow.png")
plt.close(fig1)

# ── 2. TR COMPARISON CHART ────────────────────────────────────────────────────
TR_ALPHA   = 0.2
TR_N_BLEND = 0.2
TR_N_0     = 2
TR_LAMBDA  = 5

def simulate_tr(C_list, k_offset=5, tr_init=0.25, mean_init=0.50, m2_init=0.0):
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

tr_p1, conf_p1 = simulate_tr(C_P1)
tr_p2, conf_p2 = simulate_tr(C_P2)

fig2, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
fig2.subplots_adjust(wspace=0.06, top=0.82)

panels = [
    (axes[0], C_P1, tr_p1, conf_p1, "Consistent contributor", "#009E73"),
    (axes[1], C_P2, tr_p2, conf_p2, "Erratic contributor",    "#D55E00"),
]

for ax, C_list, tr_list, conf_list, title, col in panels:
    ax.set_title(title, fontsize=15, fontweight="bold", color=col, pad=8)
    ax.set_xlabel("Tasks completed (this type)", fontsize=12, labelpad=5)
    ax.set_xticks(tasks)
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.6, color="grey", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # contribution score — faint background bars
    ax.bar(tasks, C_list, color=col, alpha=0.12, width=0.6, zorder=1,
           label="Contribution score (each task)")

    # confidence line
    ax.plot(tasks, conf_list, color="#0072B2", linewidth=2,
            linestyle="--", marker="s", markersize=6,
            label="Confidence", zorder=3)

    # TR line — thick & prominent
    ax.plot(tasks, tr_list, color=col, linewidth=3,
            linestyle="-", marker="o", markersize=8,
            label="Task Reputation (TR)", zorder=4)

axes[0].set_ylabel("Score", fontsize=12, labelpad=5)

# single legend centred below
handles, labels = axes[0].get_legend_handles_labels()
fig2.legend(handles, labels,
            loc="upper center", ncol=3,
            fontsize=11, frameon=False,
            bbox_to_anchor=(0.5, 0.98))

# callout annotation on erratic panel
axes[1].annotate(
    "erratic contributions\nsuppress confidence\n→ TR grows slowly",
    xy=(3, tr_p2[2]), xytext=(5.2, 0.55),
    fontsize=9.5, color="#333333",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=1),
)

fig2.savefig(OUT / "pres_tr_compare.png", dpi=180, bbox_inches="tight")
print("Saved pres_tr_compare.png")
plt.close(fig2)
