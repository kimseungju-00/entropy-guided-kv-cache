"""
Fig. 5: per-layer normalized entropy phi (top) and retention ratio (bottom) of the proposed
method at eps = 0.5 for the three models.

Bars: mean over the samples of the five tasks (equal sample counts, so the mean of the task
means). Whiskers: range of the five task means. Dashed line: mean retention ratio over layers,
i.e., the uniform ratio with the same average budget.

Usage: python plot_fig5.py [out.png]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
MODELS = [("Mistral-7B-Instruct-v0.3", "Mistral-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
          ("Qwen3-8B", "Qwen3-8B")]
CFG = "M1_e0.5"
PHI, KEEP = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def load(model):
    res = json.load(open(os.path.join(ROOT, "results", "revision_phase1", f"{model}.json")))[CFG]
    phi = np.array([res[t]["layer_phi_mean"] for t in TASKS])
    keep = np.array([res[t]["layer_keep_mean"] for t in TASKS])
    return phi, keep


def panel(ax, v, color, ylim, mean_line=None):
    x = np.arange(v.shape[1]); m = v.mean(0)
    ax.bar(x, m, width=0.72, color=color, linewidth=0, zorder=2)
    ax.errorbar(x, m, yerr=[m - v.min(0), v.max(0) - m], fmt="none", ecolor=INK, elinewidth=0.6,
                capsize=1.2, capthick=0.6, zorder=3)
    if mean_line is not None:
        ax.axhline(mean_line, color=INK, linestyle="--", linewidth=0.8, zorder=4)
    ax.set_ylim(*ylim); ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_xticks([t for t in x if t % 4 == 0])
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)


def main(out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.9), sharey="row",
                             gridspec_kw={"width_ratios": [32, 32, 36]})
    for j, (model, name) in enumerate(MODELS):
        phi, keep = load(model)
        panel(axes[0, j], phi, PHI, (0, 1.05))
        mk = keep.mean()
        panel(axes[1, j], keep, KEEP, (0, 0.55), mean_line=mk)
        axes[1, j].text(0.02, 0.97, f"- - mean {mk:.3f}", transform=axes[1, j].transAxes, ha="left", va="top",
                        fontsize=6.5, color=INK)
        axes[0, j].set_title(name, fontsize=8.5, color=INK)
        axes[1, j].set_xlabel("Layer", fontsize=7.5, color=MUTED)
    axes[0, 0].set_ylabel("Normalized entropy φ", fontsize=7.5, color=MUTED)
    axes[1, 0].set_ylabel("Retention ratio", fontsize=7.5, color=MUTED)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    fig.savefig(out, dpi=300, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "figures", "fig5_layer_allocation.png"))
