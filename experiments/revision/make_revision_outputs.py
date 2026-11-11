"""
Figures 3-4 and Tables 6, A1-A3 of the revised manuscript, from the revision results.

All comparisons use the matching rule of src/eval/matching.py (isotonic fit, linear
interpolation, all points, no extrapolation; paired bootstrap CIs).

Writes:
  results/figures/fig3_budget_score.png   accuracy vs realized budget, all methods
  results/figures/fig4_memory.png         memory reduction vs each baseline at matched accuracy
  results/tables/table6_memory.{json,md}  memory at matched accuracy, all points, with CIs
  results/tables/tableA_<model>.{json,md} per-task accuracy, all methods and budgets
The JSON tables ({"headers", "rows", "bold"}) hold the cells of the manuscript tables.

Usage: python make_revision_outputs.py [--boot 2000]
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from src.eval.matching import load_curves, matched, TASKS

MODELS = [("Mistral-7B-Instruct-v0.3", "Mistral-7B"), ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
          ("Qwen3-8B", "Qwen3-8B")]
# KV cache size at 8,000 tokens with the full cache: layers x KV heads x head dim x (K, V) x 2 bytes
FULL_GB = {"Mistral-7B": 32 * 8 * 128 * 2 * 2 * 8000 / 2 ** 30, "Llama-3.1-8B": 32 * 8 * 128 * 2 * 2 * 8000 / 2 ** 30,
           "Qwen3-8B": 36 * 8 * 128 * 2 * 2 * 8000 / 2 ** 30}
BASELINES = ["SnapKV", "PyramidKV", "AdaSnapKV", "CAKE"]
LABEL = {"AdaSnapKV": "Ada-SnapKV", "Uniform": "Uniform (same press)"}
# entity colors (validated categorical order), fixed per method across figures
COLOR = {"Proposed": "#2a78d6", "Uniform": "#eb6834", "SnapKV": "#1baf7a", "PyramidKV": "#eda100",
         "AdaSnapKV": "#e87ba4", "CAKE": "#008300"}
MARK = {"Proposed": "o", "Uniform": "D", "SnapKV": "s", "PyramidKV": "^", "AdaSnapKV": "v", "CAKE": "P"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
OLD = {"FullKV": "FullKV", "Random": "Random", "StreamingLLM": "StreamingLLM", "Knorm": "Knorm"}


def eps(cfg): return cfg.split("_e")[1]


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)


def old_perf(model):
    p = os.path.join(ROOT, "results", "kvpress_performance", f"{model}_perf.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def full_score(model):
    fk = old_perf(model).get("FullKV", {}).get("FullKV", {}).get("tasks")
    return np.mean([fk[t]["score"] for t in TASKS]) if fk else None


def fig3(curves):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6))
    for ax, (model, name) in zip(axes, MODELS):
        c = curves[model]
        for m in ["SnapKV", "PyramidKV", "AdaSnapKV", "CAKE", "Uniform", "Proposed"]:
            if m not in c: continue
            pts = c[m]
            ax.plot([p.budget for p in pts], [p.score for p in pts], marker=MARK[m], color=COLOR[m],
                    lw=1.8 if m == "Proposed" else 1.1, ms=3.6 if m == "Proposed" else 2.8,
                    label=LABEL.get(m, m), zorder=3 if m == "Proposed" else 2)
        fs = full_score(model)
        if fs is not None:
            ax.axhline(fs, color=MUTED, ls=":", lw=0.9, label="FullKV")
        ax.set_title(name, fontsize=8.5, color=INK); ax.set_xlabel("Realized retention ratio", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, 0.53); style(ax)
    axes[0].set_ylabel("Mean score (5 tasks)", fontsize=7.5, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=6.5, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.08, 1, 1))
    out = os.path.join(ROOT, "results", "figures", "fig3_budget_score.png")
    fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)


def fig4(res):
    lo, hi = -100, 80
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
    off = {b: (i - 1.5) * 0.006 for i, b in enumerate(BASELINES)}
    for ax, (model, name) in zip(axes, MODELS):
        ax.axhline(0, color=INK, lw=0.7)
        for b in BASELINES:
            r = res[model].get(b)
            if r is None: continue
            pts = [p for p in r["points"] if p["save_pct"] is not None]
            x = np.array([p["budget"] for p in pts]) + off[b]; y = np.array([p["save_pct"] for p in pts])
            # intervals are truncated to the plotted range; a percentile interval may exclude the estimate
            elo = [max(0, p["save_pct"] - max(p["ci"][0], lo)) if p.get("ci") else 0 for p in pts]
            ehi = [max(0, min(p["ci"][1], hi) - p["save_pct"]) if p.get("ci") else 0 for p in pts]
            ax.errorbar(x, y, yerr=[elo, ehi], fmt=MARK[b], color=COLOR[b], ms=3.0, elinewidth=0.6, capsize=1.0, alpha=0.9,
                        label=LABEL.get(b, b), linestyle="-", lw=0.6)
        ax.set_ylim(lo, hi); ax.set_xlim(0.04, 0.46)
        ax.set_title(name, fontsize=8.5, color=INK); ax.set_xlabel("Retention ratio (proposed)", fontsize=7.5, color=MUTED)
        style(ax)
    axes[0].set_ylabel("Memory reduction (%)", fontsize=7.5, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.08, 1, 1))
    out = os.path.join(ROOT, "results", "figures", "fig4_memory.png")
    fig.savefig(out, dpi=300, facecolor="white"); print("saved", out)


def save_table(name, headers, rows, bold):
    d = os.path.join(ROOT, "results", "tables"); os.makedirs(d, exist_ok=True)
    json.dump({"headers": headers, "rows": rows, "bold": bold}, open(os.path.join(d, f"{name}.json"), "w"),
              indent=1, ensure_ascii=False)
    md = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    md += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    open(os.path.join(d, f"{name}.md"), "w").write(md); print("saved", name)


def cell(v, ci):
    if v is None: return "n.m."
    s = f"{v:+.1f}"
    return s + (f" [{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else "")


def table6(res, curves):
    headers = ["Model", "ε", "Ratio", "Score", "Memory (GB)"] + [LABEL.get(b, b) for b in BASELINES]
    rows, bold = [], []
    for model, name in MODELS:
        prop = curves[model]["Proposed"]
        for i, p in enumerate(prop):
            row = [name if i == 0 else "", eps(p.cfg), f"{p.budget:.3f}", f"{p.score:.4f}", f"{p.budget * FULL_GB[name]:.3f}"]
            for b in BASELINES:
                if b not in res[model]: row.append("—"); continue
                q = res[model][b]["points"][i]; assert q["cfg"] == p.cfg
                row.append(cell(q["save_pct"], q.get("ci")))
            rows.append(row)
        row = [f"{name} mean", "", "", "", ""]
        for b in BASELINES:
            r = res[model].get(b)
            if r is None: row.append("—"); continue
            row.append(cell(r.get("mean_save_pct"), r.get("mean_ci")) + f" ({r['n_matched']})")
        rows.append(row); bold.append(len(rows) - 1)
    save_table("table6_memory", headers, rows, bold)


def tableA(curves):
    tl = {"qasper": "Qasper", "hotpotqa": "HotpotQA", "gov_report": "GovReport", "trec": "TREC", "lcc": "LCC"}
    for model, name in MODELS:
        headers = ["Method", "Retention ratio"] + [tl[t] for t in TASKS] + ["Avg"]
        rows = []
        old = old_perf(model)
        for m in ["FullKV", "Random", "StreamingLLM", "Knorm"]:
            for k, (tag, info) in enumerate(sorted(old.get(m, {}).items(), key=lambda x: x[1].get("keep", 1.0))):
                t = info["tasks"]; s = [t[x]["score"] for x in TASKS]
                rows.append([m if k == 0 else "", f"{info.get('keep', 1.0):.3f}"] + [f"{v:.4f}" for v in s] + [f"{np.mean(s):.4f}"])
        for m in ["SnapKV", "PyramidKV", "AdaSnapKV", "CAKE", "Uniform", "Proposed"]:
            for k, p in enumerate(curves[model].get(m, [])):
                s = [p.scores[t].mean() for t in TASKS]
                lab = LABEL.get(m, m) if k == 0 else ""
                if m == "Proposed": lab = f"Proposed (ε = {eps(p.cfg)})"
                rows.append([lab, f"{p.budget:.3f}"] + [f"{v:.4f}" for v in s] + [f"{np.mean(s):.4f}"])
        save_table(f"tableA_{name}", headers, rows, [])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    curves = {m: load_curves(ROOT, m) for m, _ in MODELS}
    res = {m: {b: matched(curves[m]["Proposed"], curves[m][b], boot=a.boot) for b in BASELINES if b in curves[m]}
           for m, _ in MODELS}
    fig3(curves); fig4(res); table6(res, curves); tableA(curves)


if __name__ == "__main__":
    main()
