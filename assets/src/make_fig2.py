import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8,
    "legend.fontsize": 6.6,
})

B    = np.array([32, 64, 128, 256, 512])
MIOU = np.array([0.0147, 0.0486, 0.0931, 0.1080, 0.1080])
COV  = np.array([0.186, 0.413, 0.778, 0.994, 0.994])
FMT  = np.array([0.917, 1.000, 1.000, 1.000, 1.000])
UNIT = np.array([0.92, 1.92, 3.77, 5.42, 5.44])
GT_UNITS = 5.31

C_M, C_C, C_F, C_G = "#B85C36", "#2E7D64", "#5B52B5", "#8C8A82"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.35),
                             gridspec_kw={"width_ratios": [1.28, 1]})

a1.axvline(256, color=C_G, lw=0.7, ls=(0, (2.5, 2)), zorder=1)
a1.plot(B, MIOU, "o-", color=C_M, lw=1.4, ms=4, mfc="white", mew=1.2,
        label="STG mIoU (left)", zorder=4)
a1.set_xscale("log", base=2)
a1.set_xticks(B); a1.set_xticklabels([str(b) for b in B])
a1.set_xlabel("max new tokens")
a1.set_ylabel("STG mIoU", color=C_M)
a1.set_ylim(0, 0.128); a1.tick_params(axis="y", colors=C_M)
a1.set_xlim(27, 610)

b1 = a1.twinx()
b1.plot(B, COV, "s--", color=C_C, lw=1.2, ms=3.4, mfc="white", mew=1.0,
        label="output coverage (right)", zorder=3)
b1.plot(B, FMT, "^:", color=C_F, lw=1.2, ms=3.8, mfc="white", mew=1.0,
        label="format-valid rate (right)", zorder=3)
b1.set_ylabel("coverage / format-valid rate")
b1.set_ylim(0, 1.06)
b1.spines["top"].set_visible(False); a1.spines["top"].set_visible(False)

a1.annotate("coverage saturates", xy=(256, 0.104), xytext=(120, 0.121),
            fontsize=6.2, color=C_G, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=C_G, lw=0.6,
                            shrinkA=2, shrinkB=3))
a1.annotate("92% format-valid\nat mIoU 0.015", xy=(32, 0.0147),
            xytext=(38, 0.048), fontsize=6.2, color=C_F, ha="left",
            va="center", arrowprops=dict(arrowstyle="->", color=C_F,
                                         lw=0.6, shrinkA=2, shrinkB=3))

h1, l1 = a1.get_legend_handles_labels()
h2, l2 = b1.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, loc="lower right", bbox_to_anchor=(0.995, 0.02),
          frameon=False, handlelength=2.2, labelspacing=0.3)
a1.set_title("(a) STG performance is bounded by output budget",
             fontsize=8, pad=5)

w = 0.36
idx = np.arange(len(B))
a2.bar(idx - w / 2, UNIT, w, color=C_M, ec="none", label="boxes emitted")
a2.bar(idx + w / 2, [GT_UNITS] * len(B), w, color="none", ec=C_G, lw=0.8,
       ls="--", label="boxes required")
a2.set_xticks(idx); a2.set_xticklabels([str(b) for b in B])
a2.set_xlabel("max new tokens"); a2.set_ylabel("boxes per sample")
a2.set_ylim(0, 6.6); a2.spines["top"].set_visible(False)
a2.spines["right"].set_visible(False)

c2 = a2.twinx()
ratio = MIOU / COV
c2.plot(idx, ratio, "o-", color=C_C, lw=1.4, ms=4, mfc="white", mew=1.2,
        label="mIoU / coverage")
c2.set_ylabel("per-box IoU", color=C_C)
c2.set_ylim(0, 0.20); c2.tick_params(axis="y", colors=C_C)
c2.spines["top"].set_visible(False)
c2.axhline(ratio[1:].mean(), color=C_C, lw=0.6, ls=(0, (1.5, 2)))
c2.text(4.38, ratio[1:].mean() + 0.010, "constant 0.11", fontsize=6.2,
        color=C_C, ha="right")

h3, l3 = a2.get_legend_handles_labels()
h4, l4 = c2.get_legend_handles_labels()
a2.legend(h3 + h4, l3 + l4, loc="upper left", bbox_to_anchor=(0.0, 1.0),
          frameon=False,
          handlelength=2.0, labelspacing=0.3)
a2.set_title("(b) Quality is constant; coverage is everything",
             fontsize=8, pad=5)

fig.tight_layout(pad=0.35, w_pad=1.8)
fig.savefig("/mnt/user-data/outputs/spiral-medvidu/assets/fig2_token_budget.pdf", bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig("/mnt/user-data/outputs/spiral-medvidu/assets/fig2_token_budget.png", dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
print("ok")
