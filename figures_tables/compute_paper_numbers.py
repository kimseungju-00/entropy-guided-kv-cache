"""
Every number quoted in the text of the revised manuscript, computed from the results.

Run from the project root (the paper used --boot 10000; the seed is fixed, so the output is
reproduced exactly):
  python figures_tables/compute_paper_numbers.py [results/tables/paper_numbers.json] [--boot 10000]

Each key holds the formatted text that appears in the manuscript, for example
"+1.08 points (95% CI [+0.23, +1.95]; 10 matched points)". The manuscript writes the interval
in square brackets after the estimate and lists the numbers of matched points in Tables 7 and 8
(make_comparison_tables.py).
"""
import json, os, sys, argparse
import numpy as np
REL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REL)
from src.eval.matching import load_curves, matched, matched_score, TASKS, REVISION, isotonic, budget_at

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default=os.path.join(REL, "results", "tables", "paper_numbers.json"))
ap.add_argument("--boot", type=int, default=10000)
args = ap.parse_args()
num = json.load(open(args.out)) if os.path.exists(args.out) else {}
MODELS = {"mistral": "Mistral-7B-Instruct-v0.3", "llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen3-8B"}
curves = {k: load_curves(REL, m, list(REVISION) + ["RoleWeighted"]) for k, m in MODELS.items()}
set_ = {}


def pct(r):
    """'x% (95% CI [a, b]; n points)' for a mean saving, sign kept."""
    if "mean_save_pct" not in r: return None
    s = f"{r['mean_save_pct']:.1f}%"
    if "mean_ci" in r: s += f" (95% CI [{r['mean_ci'][0]:.1f}%, {r['mean_ci'][1]:.1f}%]"
    else: s += " (no CI: means only"
    return s + f"; {r['n_matched']} matched points)"


def comp(model, a, b):
    c = curves[model]
    if a in c and b in c and len(c[b]) >= 2:
        return matched(c[a], c[b], boot=args.boot)
    return None


def pts(r):
    """'+x points (95% CI [a, b]; n points)' for a mean score difference at matched budget."""
    if "mean_diff_pts" not in r: return None
    s = f"{r['mean_diff_pts']:+.2f} points"
    if "mean_ci" in r: s += f" (95% CI [{r['mean_ci'][0]:+.2f}, {r['mean_ci'][1]:+.2f}]"
    else: s += " (no CI: means only"
    return s + f"; {r['n_matched']} matched points)"


def comp_score(model, a, b):
    c = curves[model]
    if a in c and b in c and len(c[b]) >= 2:
        return matched_score(c[a], c[b], boot=args.boot)
    return None


def score_diff(model, cfg_a, method_a, cfg_b, method_b):
    """Paired difference of mean scores (points) with bootstrap CI, per-sample data only."""
    c = curves[model]
    try:
        pa = next(p for p in c[method_a] if p.cfg == cfg_a); pb = next(p for p in c[method_b] if p.cfg == cfg_b)
    except (KeyError, StopIteration):
        return None
    if pa.scores is None or pb.scores is None: return None
    rng = np.random.default_rng(0); n = {t: len(pa.scores[t]) for t in TASKS}
    d = [pa.mean_score(idx) - pb.mean_score(idx) for idx in
         ({t: rng.integers(0, n[t], n[t]) for t in TASKS} for _ in range(args.boot))]
    return f"{(pa.score - pb.score) * 100:.2f} points, 95% CI [{np.percentile(d, 2.5) * 100:.2f}, {np.percentile(d, 97.5) * 100:.2f}]"


# budget range of the proposed method
bud = [p.budget for k in MODELS for p in curves[k].get("Proposed", []) if p.scores is not None]
if bud: set_["prop_budget_range"] = f"realized retention ratios {min(bud):.2f}–{max(bud):.2f}"

set_["diff_mistral_ada"] = score_diff("mistral", "M1_e0.5", "Proposed", "AdaSnap_0.2", "AdaSnapKV")
set_["diff_qwen_snap"] = score_diff("qwen", "M1_e0.4", "Proposed", "Snap_0.2", "SnapKV")

for m in MODELS:
    for base, key in [("SnapKV", "snap"), ("PyramidKV", "pyr"), ("AdaSnapKV", "ada")]:
        r = comp(m, "Proposed", base)
        if r: set_[f"mem_{m}_{key}"] = pct(r)
    for base, key in [("Uniform", "uni"), ("CAKE", "cake")]:
        r = comp(m, "Proposed", base)
        if r: set_[f"{key}_{m}"] = pct(r)
        r = comp_score(m, "Proposed", base)
        if r: set_[f"{key}_{m}_acc"] = pts(r)
    # CAKE vs uniform, and the CAKE gap decomposition (layer allocation x token score)
    for a, b, key in [("CAKE", "Uniform", "cakeuni"),
                      ("Proposed+CAKEscore", "CAKE", "decomp_alloc_cakescore"),
                      ("Proposed", "CAKEalloc+meanscore", "decomp_alloc_meanscore"),
                      ("Proposed+CAKEscore", "Proposed", "decomp_score_prop"),
                      ("CAKE", "CAKEalloc+meanscore", "decomp_score_cake"),
                      ("CAKEalloc+meanscore", "Uniform", "decomp_cakealloc_uni"),
                      ("EntropyNorm", "Proposed", "norm_prop"), ("EntropyNorm", "Uniform", "norm_uni"),
                      ("EntropyNorm", "CAKE", "norm_cake"),
                      ("EntropyNorm", "CAKEalloc+meanscore", "norm_cakealloc"),
                      ("EntropyNorm-a4", "Proposed", "norma4_prop"),
                      ("EntropyNorm-a4", "CAKEalloc+meanscore", "norma4_cakealloc"),
                      ("Proposed-logT", "Proposed", "logT")]:
        r = comp_score(m, a, b)
        if r: set_[f"{key}_{m}_acc"] = pts(r)
    r = comp(m, "Proposed+AdaKV", "AdaSnapKV")
    if r: set_[f"combo_{m}"] = pct(r)
    r = comp_score(m, "Proposed+AdaKV", "AdaSnapKV")
    if r: set_[f"combo_{m}_acc"] = pts(r)
    # press decomposition: observation window (32 vs 64) x implementation (ours vs kvpress SnapKV)
    for a, b, key in [("Proposed", "SnapKV", "propsnap"), ("Uniform", "SnapKV", "press_uni_snap"), ("SnapKV-w32", "SnapKV", "press_snap32_snap"),
                      ("Uniform", "SnapKV-w32", "press_uni_snap32"), ("Uniform-w64", "SnapKV", "press_uni64_snap"),
                      ("Uniform", "Uniform-w64", "press_uni_uni64"), ("Proposed", "SnapKV-w32", "press_prop_snap32")]:
        r = comp_score(m, a, b)
        if r: set_[f"{key}_{m}_acc"] = pts(r)
    r = comp(m, "Uniform", "SnapKV")
    if r: set_[f"press_uni_snap_{m}"] = pct(r)
    # accuracy of the uniform ratio at 0.2 for observation windows of 16, 32, 64 and 128 tokens
    win = [next((p.score for p in curves[m].get(k, []) if p.cfg.endswith("_0.2")), None)
           for k in ["Uniform-w16", "Uniform", "Uniform-w64", "Uniform-w128"]]
    if None not in win: set_[f"win_{m}"] = ", ".join(f"{w * 100:.2f}" for w in win[:3]) + f", and {win[3] * 100:.2f}"

r = comp("qwen", "Proposed", "SnapKV")
if r:
    sv = [p["save_pct"] for p in r["points"] if p["save_pct"] is not None]
    if sv: set_["mem_qwen_snap_range"] = f"{min(sv):.1f}% to {max(sv):.1f}%"
    cis = [p["ci"] for p in r["points"] if p.get("ci")]
    if cis: set_["mem_qwen_snap_ci0"] = f"{sum(lo <= 0 <= hi for lo, hi in cis)} of {len(cis)} points"

# final method vs role-weighted variant (earlier results, means only)
parts = []
NAME = {"mistral": "Mistral-7B", "llama": "Llama-3.1-8B", "qwen": "Qwen3-8B"}
for m in MODELS:
    rr = comp(m, "Proposed", "RoleWeighted")
    if rr and "mean_save_pct" in rr:
        v = rr["mean_save_pct"]
        parts.append(f"{abs(v):.1f}% {'less' if v > 0 else 'more'} on {NAME[m]}")
if parts: set_["rw_vs_prop"] = ", ".join(parts)

# best low-budget point vs uniform (Llama), and the spread of the realized budget at eps = 0.5
r = comp_score("llama", "Proposed", "Uniform")
if r:
    p0 = next((p for p in r["points"] if p["cfg"] == "M1_e0.3" and "ci" in p), None)
    if p0: set_["uni_llama_lowpoint"] = (f"{p0['diff_pts']:+.2f} points at a retention ratio of {p0['budget']:.2f}, "
                                        f"95% CI [{p0['ci'][0]:+.2f}, {p0['ci'][1]:+.2f}]")
sds, spreads = [], []
for k in MODELS:
    p = next((p for p in curves[k].get("Proposed", []) if p.cfg == "M1_e0.5" and p.budgets is not None), None)
    if p is None: continue
    sds += [p.budgets[t].std() for t in TASKS]
    tm = [p.budgets[t].mean() for t in TASKS]; spreads.append(max(tm) - min(tm))
if sds: set_["budget_sd_max"] = f"{max(sds):.3f}"; set_["budget_task_spread_max"] = f"{max(spreads):.2f}"

# per-layer keep-ratio range at eps = 0.5
for k, m in MODELS.items():
    path = os.path.join(REL, "results", "revision_phase1", f"{m}.json")
    if not os.path.exists(path): continue
    res = json.load(open(path)).get("M1_e0.5", {})
    lk = [res[t]["layer_keep_mean"] for t in TASKS if t in res and res[t].get("layer_keep_mean")]
    if len(lk) == len(TASKS):
        lk = np.mean(lk, axis=0); set_[f"layer_range_{k}"] = f"{lk.min():.2f} to {lk.max():.2f}"

# contrast between layers: coefficient of variation of the mean per-layer retention ratio
def layer_cv(model_file, cfg):
    path = os.path.join(REL, "results", "revision_phase1", model_file)
    if not os.path.exists(path): return None
    res = json.load(open(path)).get(cfg, {})
    lk = [res[t].get("layer_keep_mean") for t in TASKS if t in res]
    if len(lk) < len(TASKS) or any(x is None for x in lk): return None
    a = np.mean(lk, axis=0); return float(a.std() / a.mean())

for k in ["mistral", "llama"]:
    lo, hi = layer_cv(f"{MODELS[k]}.json", "M1_e0.3"), layer_cv(f"{MODELS[k]}.json", "M1_e3.0")
    if lo is not None and hi is not None: set_[f"cv_prop_{k}"] = f"{lo:.2f} at ε = 0.3 to {hi:.2f} at ε = 3.0"
cv = [layer_cv(f"{MODELS[k]}_cake.json", f"CAKE_{r}") for k in ["mistral", "llama"] for r in [0.1, 0.15, 0.2, 0.3, 0.4]]
cv = [c for c in cv if c is not None]
if cv: set_["cv_cake_range"] = f"{min(cv):.2f} and {max(cv):.2f}"

# sensitivity of the mean reduction relative to SnapKV to the matching rule (point estimates)
def crossings(xs, ys, t):
    return [xs[i - 1] + (xs[i] - xs[i - 1]) * (t - ys[i - 1]) / (ys[i] - ys[i - 1])
            for i in range(1, len(xs)) if (ys[i - 1] - t) * (ys[i] - t) <= 0 and ys[i] != ys[i - 1]]

def rule_savings(prop, base):
    px = np.array([p.budget for p in prop]); py = np.array([p.score for p in prop]); pf = isotonic(py)
    xs = np.array([p.budget for p in base]); ys = np.array([p.score for p in base]); fit = isotonic(ys)
    out = {"iso": [], "both": [], "first": [], "last": []}
    for x, y, yf in zip(px, py, pf):
        b = budget_at(xs, fit, y)
        if b: out["iso"].append((b - x) / b * 100)
        b = budget_at(xs, fit, yf)
        if b: out["both"].append((b - x) / b * 100)
        c = crossings(xs, ys, y)
        if c:
            out["first"].append((min(c) - x) / min(c) * 100); out["last"].append((max(c) - x) / max(c) * 100)
    return out

for k in MODELS:
    r = rule_savings(curves[k]["Proposed"], curves[k]["SnapKV"])
    m = [np.mean(v) for v in r.values() if v]
    set_[f"sens_{k}"] = f"{min(m):.1f}% to {max(m):.1f}%" if max(m) - min(m) >= 0.05 else f"{m[0]:.1f}% under all of them"
qp = next(p for p in curves["qwen"]["Proposed"] if abs(p.score - 0.1669) < 5e-4)
xs = np.array([p.budget for p in curves["qwen"]["SnapKV"]]); ys = np.array([p.score for p in curves["qwen"]["SnapKV"]])
c = crossings(xs, ys, qp.score); b = budget_at(xs, isotonic(ys), qp.score)
set_["sens_qwen_point_first"] = f"{(min(c) - qp.budget) / min(c) * 100:+.1f}%"
set_["sens_qwen_point_iso"] = f"{(b - qp.budget) / b * 100:+.1f}%"
set_["sens_qwen_point_detail"] = f"SnapKV ratio {min(c):.3f} (first) vs {b:.3f} (isotonic), crossings {[round(x, 3) for x in c]}"
f = isotonic([p.score for p in curves["qwen"]["PyramidKV"]])
set_["pyr_qwen_range"] = f"accuracies of {f[0]:.3f} to {f[-1]:.3f}"

# baselines with the observation window of the proposed press
for k in MODELS:
    for base, tag in (("PyramidKV-w32", "pyr"), ("AdaSnapKV-w32", "ada")):
        r = comp(k, "Proposed", base)
        if r: set_[f"w32_prop_{tag}_{k}"] = pct(r)
        r = comp_score(k, "Proposed", base)
        if r: set_[f"w32_prop_{tag}_{k}_acc"] = pts(r)
    for a, b, tag in (("PyramidKV-w32", "PyramidKV", "pyr"), ("AdaSnapKV-w32", "AdaSnapKV", "ada")):
        r = comp_score(k, a, b)
        if r: set_[f"w32_{tag}_self_{k}_acc"] = pts(r)
    r = comp_score(k, "Proposed+AdaKV", "AdaSnapKV-w32")
    if r: set_[f"w32_combo_{k}_acc"] = pts(r)

# per-layer discarded mass versus phi (analyze_layer_delta.py)
ld_path = os.path.join(REL, "results", "tables", "layer_delta.json")
if os.path.exists(ld_path):
    ld = json.load(open(ld_path)); keys = list(MODELS)
    three = lambda name: ", ".join(f"{ld[MODELS[k]][name]['pearson']:.2f}" for k in keys[:2]) + f", and {ld[MODELS[keys[2]]][name]['pearson']:.2f}"
    set_["ld_phi_H"] = three("phi_H")
    set_["ld_phi_delta"] = three("phi_delta_0.2")
    set_["ld_H_delta"] = three("H_delta_0.2")
    set_["ld_phi_delta_qwen_wo"] = f"{ld[MODELS['qwen']]['without_near_uniform_layers']['phi_delta_0.2']['pearson']:.2f}"
    for k in keys:
        t = ld[MODELS[k]]["rule_vs_uniform"]["0.5"]
        set_[f"ld_rule_{k}"] = (f"{t['delta_rule']:.4f} under the rule and {t['delta_uniform']:.4f} under the uniform ratio "
                                f"(difference {t['diff']:+.4f}, 95% CI [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}])")

# distribution-level effect of the rule and normalization properties (analyze_allocation_benefit.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_benefit_numbers import benefit_keys
set_.update(benefit_keys())

changed = {k: v for k, v in set_.items() if v is not None}
num.update(changed)
json.dump(num, open(args.out, "w"), indent=2, ensure_ascii=False)
print(f"{len(changed)} numeric keys written to {args.out}")
for k, v in sorted(changed.items()): print(f"  {k}: {v}")
