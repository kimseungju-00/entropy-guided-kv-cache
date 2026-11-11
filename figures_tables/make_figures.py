"""
Generate the paper figures from the result JSON files.

Run from the project root:
  python make_figures.py

Reads:
  results/kvpress_theory/{model}_delta.json        (TV=delta, delta-entropy)
  results/kvpress_theory/{model}_layerent.json     (layer entropy)
  results/kvpress_performance/{model}_perf.json     (accuracy vs budget)
  results/kvpress_memory_m1/{model}_m1mem_final.json (matched-performance memory)
  results/revision_phase1/{model}.json              (per-layer phi / keep ratio, Figure 5)
Writes: figures/*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json, os, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['font.size'] = 11
os.makedirs("figures", exist_ok=True)

MODELS = {
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
}
C = {"prop": "#2E7D32", "snap": "#00838F", "pyr": "#E65100", "full": "#1565C0"}
CC = {"Mistral-7B": "#2E7D32", "Llama-3.1-8B": "#1565C0", "Qwen3-8B": "#6A1B9A"}
MK = {"Mistral-7B": "o", "Llama-3.1-8B": "s", "Qwen3-8B": "^"}

def load(path):
    if not os.path.exists(path):
        print(f"  [skip] not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)

# Figure 1: delta-entropy correlation
def fig_delta_entropy():
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ok = False
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_theory/{mkey}_delta.json")
        if d is None: continue
        res = d.get("results", d)
        rhos = sorted([float(k) for k in res.keys() if k not in ("window",)])
        corr = [res[str(r)]["pearson_delta_H"] for r in rhos]
        ax.plot(rhos, corr, marker=MK[mname], label=mname, color=CC[mname], lw=2, ms=7); ok = True
    if not ok: print("  [fig1 skipped] no delta files"); plt.close(); return
    ax.set_xlabel("Retention ratio ρ (fraction kept)")
    ax.set_ylabel("Pearson correlation (δ vs entropy H)")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); os.makedirs("results/figures", exist_ok=True); fig.savefig("results/figures/fig1_delta_entropy.png", dpi=200, bbox_inches="tight")
    print("saved results/figures/fig1_delta_entropy.png")

# Figure 2: layer entropy heterogeneity
def fig_layer_entropy():
    data = {}
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_theory/{mkey}_layerent.json")
        if d is None: continue
        data[mname] = (d["layer_entropy"], d["ratio"])
    if not data: print("  [fig2 skipped] no layerent files"); return
    fig, axes = plt.subplots(1, len(data), figsize=(4.7*len(data), 3.8))
    if len(data) == 1: axes = [axes]
    for ax, (mname, (ent, ratio)) in zip(axes, data.items()):
        x = np.arange(len(ent))
        ax.bar(x, ent, color=CC[mname], alpha=0.8, width=0.9)
        ax.axhline(np.mean(ent), color="black", ls="--", lw=1, alpha=0.6)
        ax.set_title(mname, fontsize=11)   # panel label only; the caption describes the figure
        ax.set_xlabel("Layer index")
        if ax is axes[0]: ax.set_ylabel("Mean attention entropy")
    fig.tight_layout(); os.makedirs("results/figures", exist_ok=True); fig.savefig("results/figures/fig2_layer_entropy.png", dpi=200, bbox_inches="tight")
    print("saved results/figures/fig2_layer_entropy.png")

# Figure 3: accuracy vs budget
def fig_budget_score():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4)); ok = False
    axmap = {m: axes[i] for i, m in enumerate(MODELS.values())}
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_performance/{mkey}_perf.json")
        if d is None: continue
        ax = axmap[mname]; ok = True
        def curve(method_key, is_prop=False):
            pts = []
            if method_key not in d: return pts
            for tag, info in d[method_key].items():
                tasks = info["tasks"]
                score = np.mean([tasks[t]["score"] for t in tasks])
                if is_prop:
                    buds = [tasks[t]["budget"] for t in tasks if tasks[t].get("budget")]
                    bud = np.mean(buds) if buds else info.get("epsilon", 0)
                else:
                    bud = info.get("keep", 0)
                pts.append((bud, score))
            return sorted(pts)
        for key, lab, col, mk_ in [("SnapKV", "SnapKV", C["snap"], "s"),
                                   ("PyramidKV", "PyramidKV", C["pyr"], "^"),
                                   ("Proposed_M1", "Proposed", C["prop"], "o")]:
            pts = curve(key, key == "Proposed_M1")
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk_, label=lab, color=col, lw=2, ms=6)
        if "FullKV" in d:
            fk = d["FullKV"].get("FullKV", {}).get("tasks", {})
            if fk:
                fs = np.mean([fk[t]["score"] for t in fk])
                ax.axhline(fs, color=C["full"], ls=":", lw=1.5, label="FullKV", alpha=0.7)
        ax.set_title(mname, fontsize=11); ax.set_xlabel("KV budget (fraction kept)")
        if ax is axes[0]: ax.set_ylabel("LongBench score (5-task avg)")
        ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
    if not ok: print("  [fig3 skipped] no perf files"); plt.close(); return
    fig.tight_layout(); fig.savefig("figures/fig_budget_score.png", dpi=200, bbox_inches="tight")
    print("saved figures/fig_budget_score.png")

# Figure 4: matched-performance memory
def fig_memory():
    data = {}
    for mkey, mname in MODELS.items():
        d = load(f"results/kvpress_memory_m1/{mname}_m1mem_final.json")
        if d is None: continue
        # all matched points (rule: src/eval/matching.py); unmatched points have no SnapKV budget
        pts = [p for p in d["points"] if p["save_pct"] is not None]
        if not pts: continue
        data[mname] = {"perf": [p["score"] for p in pts], "prop": [p["m1_gb"] for p in pts],
                       "snap": [p["snap_gb"] for p in pts], "save": [p["save_pct"] for p in pts],
                       "avg": np.mean([p["save_pct"] for p in pts])}
    if not data: print("  [fig4 skipped] no memory files"); return
    fig, axes = plt.subplots(1, len(data), figsize=(4.4*len(data), 4))
    if len(data) == 1: axes = [axes]
    for ax, (mname, dd) in zip(axes, data.items()):
        x = np.arange(len(dd["perf"])); w = 0.35
        ax.bar(x-w/2, dd["snap"], w, label="SnapKV", color="#9E9E9E")
        ax.bar(x+w/2, dd["prop"], w, label="Proposed", color=C["prop"])
        for i, (s, sv) in enumerate(zip(dd["snap"], dd["save"])):
            ax.text(i, s+max(dd["snap"])*0.01, f"{-sv:+.0f}%", ha="center", fontsize=9, color=C["prop"], fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"{p:.4f}" for p in dd["perf"]])
        ax.set_xlabel("Matched score"); ax.set_title(f"{mname}\n(avg {-dd['avg']:+.1f}%)", fontsize=11)
        if ax is axes[0]: ax.set_ylabel("KV cache memory (GB)")
        ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, max(dd["snap"])*1.28)
    fig.suptitle("KV cache memory at matched performance: Proposed vs SnapKV (all matched points)", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig("figures/fig_memory.png", dpi=200, bbox_inches="tight")
    print("saved figures/fig_memory.png")

# Figure 5: per-layer phi and keep ratio of the proposed method
TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
def fig_layer_budget(cfg="M1_e0.5"):
    data = {}
    for mkey, mname in MODELS.items():
        d = load(f"results/revision_phase1/{mkey}.json")
        if d is None or cfg not in d: continue
        r = d[cfg]
        if any(t not in r or not r[t].get("layer_keep_mean") for t in TASKS): continue
        data[mname] = (np.array([r[t]["layer_phi_mean"] for t in TASKS]),
                       np.array([r[t]["layer_keep_mean"] for t in TASKS]))
    if not data: print("  [fig5 skipped] no per-layer results"); return
    fig, axes = plt.subplots(2, len(data), figsize=(4.7*len(data), 6), sharex="col")
    axes = np.array(axes).reshape(2, -1)
    for j, (mname, (phi, keep)) in enumerate(data.items()):
        for i, (v, lab) in enumerate([(phi, "Normalized entropy φ"), (keep, "Retention ratio")]):
            ax = axes[i, j]; x = np.arange(v.shape[1]); m = v.mean(0)
            # bar = mean over the five tasks, whisker = range across tasks (input dependence)
            ax.bar(x, m, width=0.8, color=CC[mname], alpha=0.85)
            ax.errorbar(x, m, yerr=[m - v.min(0), v.max(0) - m], fmt="none", ecolor="#424242",
                        elinewidth=0.8, capsize=0)
            if i == 1:
                ax.axhline(m.mean(), color="black", ls="--", lw=1, alpha=0.6)
                ax.text(0.02, 0.97, f"dashed: mean {m.mean():.2f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=9)
                ax.set_ylim(0, 0.52); ax.set_xlabel("Layer index")
            else:
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{mname}  (keep {keep.mean(0).min():.2f}–{keep.mean(0).max():.2f})", fontsize=11)
            if j == 0: ax.set_ylabel(lab)
            ax.grid(axis="y", alpha=0.3)
    eps = cfg.split("_e")[-1]
    fig.suptitle(f"Per-layer normalized entropy and retention ratio (ε = {eps}; whiskers: range over tasks)",
                 fontsize=12, y=1.01)
    fig.tight_layout(); fig.savefig("figures/fig_layer_budget.png", dpi=200, bbox_inches="tight")
    print("saved figures/fig_layer_budget.png")

if __name__ == "__main__":
    print("=== generating figures ===")
    fig_delta_entropy()
    fig_layer_entropy()
    fig_budget_score()
    fig_memory()
    fig_layer_budget()
    print("=== done. see figures/ ===")