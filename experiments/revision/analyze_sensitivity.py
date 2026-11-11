"""
Sensitivity of the mean memory reduction relative to SnapKV to the matching rule (point estimates).

Rules: the rule of Table 6 (isotonic fit of the baseline), isotonic fit of both curves, and the
first and the last crossing of the unsmoothed baseline curve. Writes results/tables/matching_sensitivity.json.
"""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from src.eval.matching import load_curves, isotonic, budget_at

MODELS = ["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "Qwen3-8B"]


def crossings(xs, ys, t):
    return [xs[i - 1] + (xs[i] - xs[i - 1]) * (t - ys[i - 1]) / (ys[i] - ys[i - 1])
            for i in range(1, len(xs)) if (ys[i - 1] - t) * (ys[i] - t) <= 0 and ys[i] != ys[i - 1]]


def rule_savings(prop, base):
    px = np.array([p.budget for p in prop]); py = np.array([p.score for p in prop]); pf = isotonic(py)
    xs = np.array([p.budget for p in base]); ys = np.array([p.score for p in base]); fit = isotonic(ys)
    out = {"isotonic_baseline": [], "isotonic_both": [], "first_crossing": [], "last_crossing": []}
    for x, y, yf in zip(px, py, pf):
        b = budget_at(xs, fit, y)
        if b: out["isotonic_baseline"].append((b - x) / b * 100)
        b = budget_at(xs, fit, yf)
        if b: out["isotonic_both"].append((b - x) / b * 100)
        c = crossings(xs, ys, y)
        if c:
            out["first_crossing"].append((min(c) - x) / min(c) * 100)
            out["last_crossing"].append((max(c) - x) / max(c) * 100)
    return out


res = {}
for m in MODELS:
    curves = load_curves(ROOT, m)
    r = rule_savings(curves["Proposed"], curves["SnapKV"])
    res[m] = {k: {"mean_save_pct": float(np.mean(v)), "n": len(v)} for k, v in r.items() if v}
    print(m, {k: round(v["mean_save_pct"], 1) for k, v in res[m].items()})
out = os.path.join(ROOT, "results", "tables", "matching_sensitivity.json")
json.dump(res, open(out, "w"), indent=1); print("saved", out)
