"""
Distribution-level effect of the per-layer rule and properties of the normalization in equation (6).

Uses the measurements of measure_layer_delta.py (nothing is evicted, so delta at any allocation
is computed afterwards from the per-layer grid). All intervals are paired bootstrap 95% CIs over
samples within tasks; "pooled" is the equal-weight mean over the three models.

Reports, for each model and epsilon, at the same per-sample mean retention ratio:
  - mean and worst-layer discarded mass under the rule and under a uniform ratio
  - budget of the uniform ratio that reaches the rule's mean discarded mass
  - rule, uniform, and the best per-sample allocation given the same grid (share of the attainable
    reduction captured), and a fixed per-model profile computed from the same samples
  - the same rule with the head entropies normalized by log T, scaled to the same mean ratio
Normalization properties: rank agreement of phi with the layer mean entropy, how often a low-entropy
layer receives a high phi, dependence on context length within a task, and contrast between layers.
Score comparisons at matched budget are repeated with an equal-weight pooled estimate.
Budget variability: within-task SD of the realized per-sample retention ratio of the proposed method
at epsilon 0.5 and of the log T variant at the configuration with the nearest mean budget.

Usage: python experiments/revision/analyze_allocation_benefit.py [--boot 10000]
Output: results/tables/allocation_benefit.json
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from src.eval.matching import load_curves, _gains, TASKS

MODELS = ["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "Qwen3-8B"]
EPS = [0.2, 0.3, 0.5, 1.0, 2.0]
RMIN, RMAX = 0.05, 0.50
GRID = np.linspace(RMIN, RMAX, 181)
LAM_GRID = np.linspace(RMIN, RMAX, 46)


def load(m):
    d = json.load(open(os.path.join(ROOT, "results", "revision_phase1", f"{m}_layerdelta.json")))
    per = []
    for t in TASKS:
        r = d[t]
        per.append(dict(phi=np.array([x["phi"] for x in r]), H=np.array([x["H"] for x in r]),
                        logT=np.array([x["logT"] for x in r]), delta=np.array([x["delta"] for x in r])))
    return np.array(d["_meta"]["rhos"]), per


def interp(rh, D, r):
    S, L, _ = D.shape
    i = np.clip(np.searchsorted(rh, r) - 1, 0, len(rh) - 2)
    w = (r - rh[i]) / (rh[i + 1] - rh[i]); l = np.arange(L); out = np.empty((S, L))
    for s in range(S): out[s] = D[s, l, i[s]] * (1 - w[s]) + D[s, l, i[s] + 1] * w[s]
    return out


def rule(stat, e):
    return np.clip(RMIN + (RMAX - RMIN) * stat ** (1 / e), RMIN, RMAX)


def scaled_rule(stat, target):
    """Same rule form with epsilon chosen per sample so that the mean ratio equals target."""
    out = np.empty_like(stat)
    for s in range(len(stat)):
        lo, hi = 0.02, 50.0
        for _ in range(40):
            e = np.sqrt(lo * hi)
            if rule(stat[s], e).mean() < target[s]: lo = e
            else: hi = e
        out[s] = rule(stat[s], e)
    return out


def best(rh, D, target):
    """Allocation minimizing the mean delta at the same mean ratio (Lagrangian over a grid)."""
    S, L, _ = D.shape
    Dg = np.stack([interp(rh, D, np.full((S, L), v)) for v in LAM_GRID], 2); out = np.empty((S, L))
    for s in range(S):
        lo, hi = 0.0, 5.0
        for _ in range(40):
            lam = (lo + hi) / 2; r = LAM_GRID[(Dg[s] + lam * LAM_GRID).argmin(1)]
            if r.mean() > target[s]: lo = lam
            else: hi = lam
        out[s] = r
    return out


def persample(rh, per, e):
    """Per-sample arrays by task for all allocations at epsilon e."""
    rs = [rule(p["phi"], e) for p in per]
    prof = np.concatenate(rs).mean(0)
    res = []
    for p, r in zip(per, rs):
        D = p["delta"]; m = r.mean(1); u = np.repeat(m[:, None], r.shape[1], 1)
        dr, du = interp(rh, D, r), interp(rh, D, u)
        Hn = np.clip(p["H"] / p["logT"][:, None], 1e-6, 1)
        lt = scaled_rule(Hn, m)
        f = np.clip(prof[None] * (m[:, None] / prof.mean()), RMIN, RMAX)
        cur = np.stack([interp(rh, D, np.full(r.shape, g)).mean(1) for g in GRID], 1)
        ueq = GRID[np.argmax(cur <= dr.mean(1)[:, None], axis=1)]
        res.append(dict(ratio=m, rule=dr.mean(1), uni=du.mean(1), rule_max=dr.max(1), uni_max=du.max(1),
                        rule_cv=dr.std(1) / dr.mean(1), uni_cv=du.std(1) / du.mean(1),
                        logt=interp(rh, D, lt).mean(1), fixed=interp(rh, D, f).mean(1),
                        best=interp(rh, D, best(rh, D, m)).mean(1), ueq=ueq,
                        rule_min_ratio=r.min(1), logt_min_ratio=lt.min(1),
                        rule_ratio_cv=r.std(1) / r.mean(1), logt_ratio_cv=lt.std(1) / lt.mean(1)))
    return res


def summarize(x):
    g = lambda k: x[k]
    return dict(ratio=g("ratio"), delta_rule=g("rule"), delta_uniform=g("uni"), delta_logt=g("logt"),
                delta_fixed=g("fixed"), delta_best=g("best"),
                mean_reduction_pct=100 * (g("uni") - g("rule")) / g("uni"),
                worst_rule=g("rule_max"), worst_uniform=g("uni_max"),
                worst_reduction_pct=100 * (g("uni_max") - g("rule_max")) / g("uni_max"),
                cv_rule=g("rule_cv"), cv_uniform=g("uni_cv"),
                captured_pct=100 * (g("uni") - g("rule")) / (g("uni") - g("best")),
                equal_delta_saving_pct=100 * (g("ueq") - g("ratio")) / g("ueq"),
                logt_vs_rule_delta_pct=100 * (g("rule") - g("logt")) / g("rule"),
                min_ratio_rule=g("rule_min_ratio"), min_ratio_logt=g("logt_min_ratio"),
                ratio_cv_rule=g("rule_ratio_cv"), ratio_cv_logt=g("logt_ratio_cv"))


def means(res, idx=None):
    keys = res[0].keys(); out = {}
    for k in keys:
        out[k] = float(np.mean([(r[k] if idx is None else r[k][i]).mean() for r, i in zip(res, idx or [None] * len(res))]))
    return out


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def norm_properties(per, rng, boot):
    out = {}
    rs, ce, tot = [], 0, 0
    for p in per:
        for s in range(len(p["phi"])):
            a, b = p["phi"][s], p["H"][s]
            rs.append(spearmanr(a, b)[0])
            lo = b <= np.quantile(b, 1 / 3); hi = a >= np.quantile(a, 2 / 3)
            ce += int((lo & hi).sum()); tot += int(lo.sum())
    out["sample_spearman_phi_H_mean"] = float(np.nanmean(rs)); out["sample_spearman_phi_H_p5"] = float(np.nanpercentile(rs, 5))
    out["low_H_third_in_top_phi_third_pct"] = 100 * ce / tot
    # dependence on context length: within task and layer, across samples
    def dep(f, idx=None):
        vals = []
        for j, p in enumerate(per):
            ii = np.arange(len(p["phi"])) if idx is None else idx[j]
            X, T = f(p)[ii], p["logT"][ii]
            if T.std() < 1e-9: continue
            Xz = (X - X.mean(0)) / (X.std(0) + 1e-12); Tz = (T - T.mean()) / T.std()
            vals.append((Xz * Tz[:, None]).mean(0))
        return float(np.nanmean(vals))
    stats = {"H": lambda p: p["H"], "H_logT": lambda p: p["H"] / p["logT"][:, None], "phi": lambda p: p["phi"]}
    for name, f in stats.items():
        est = dep(f)
        bs = [dep(f, [rng.integers(0, len(p["phi"]), len(p["phi"])) for p in per]) for _ in range(min(boot, 2000))]
        out[f"within_task_corr_logT_{name}"] = {"r": est, "ci": ci(bs)}
        X = [f(p) for p in per]
        out[f"within_task_cv_{name}"] = float(np.mean([(x.std(0) / x.mean(0)).mean() for x in X]))
    out["logT_range_by_task"] = {t: [float(p["logT"].min()), float(p["logT"].max())] for t, p in zip(TASKS, per)}
    return out


def score_pairs(curves, rng, boot):
    out = {}
    pairs = [("Proposed-logT", "Proposed"), ("Proposed", "Uniform"), ("Proposed", "PyramidKV"),
             ("Uniform", "PyramidKV"), ("Proposed", "PyramidKV-w32"), ("Uniform", "PyramidKV-w32")]
    for a, b in pairs:
        ok = [m for m in MODELS if a in curves[m] and b in curves[m]]
        est, bs = {}, {}
        for m in ok:
            A, B = curves[m][a], curves[m][b]
            est[m] = float(np.nanmean(np.array(_gains(A, B))[:, 1])); bs[m] = []
            for _ in range(boot):
                idx = {t: rng.integers(0, len(A[0].scores[t]), len(A[0].scores[t])) for t in TASKS}
                bs[m].append(np.nanmean(np.array(_gains(A, B, idx))[:, 1]))
        out[f"{a}|{b}"] = {"per_model": {m: {"diff": est[m], "ci": ci(bs[m])} for m in ok},
                            "pooled": {"diff": float(np.mean(list(est.values()))), "ci": ci(np.mean([bs[m] for m in ok], 0))}}
    return out


def budget_variability(curves, rng, boot):
    """Within-task SD of the per-sample budget: proposed at eps 0.5 vs nearest-budget log T config."""
    def sd(P, idx=None):
        return float(np.mean([(P.budgets[t] if idx is None else P.budgets[t][idx[t]]).std() for t in TASKS]))
    out = {}
    for m in MODELS:
        A = next(p for p in curves[m]["Proposed"] if p.cfg == "M1_e0.5")
        L = [p for p in curves[m]["Proposed-logT"] if p.budgets is not None]
        B = min(L, key=lambda p: abs(p.budget - A.budget))
        ratios = []
        for _ in range(boot):
            idx = {t: rng.integers(0, len(A.budgets[t]), len(A.budgets[t])) for t in TASKS}
            ratios.append(sd(B, idx) / sd(A, idx))
        out[m] = dict(proposed_cfg=A.cfg, proposed_budget=float(A.budget), proposed_sd=sd(A),
                      logt_cfg=B.cfg, logt_budget=float(B.budget), logt_sd=sd(B),
                      sd_ratio=sd(B) / sd(A), sd_ratio_ci=ci(ratios))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(); rng = np.random.default_rng(0); res = {}
    for m in MODELS:
        rh, per = load(m); r = {"normalization": norm_properties(per, rng, a.boot), "by_epsilon": {}}
        for e in EPS:
            ps = persample(rh, per, e)
            base = summarize(means(ps))
            n = [len(x["rule"]) for x in ps]
            bs = {k: [] for k in base}
            for _ in range(min(a.boot, 2000)):
                s = summarize(means(ps, [rng.integers(0, c, c) for c in n]))
                for k in s: bs[k].append(s[k])
            r["by_epsilon"][str(e)] = {k: {"est": base[k], "ci": ci(bs[k])} for k in base}
        res[m] = r
    curves = {m: load_curves(ROOT, m) for m in MODELS}
    res["scores"] = score_pairs(curves, rng, a.boot)
    res["budget_variability"] = budget_variability(curves, rng, a.boot)
    out = os.path.join(ROOT, "results", "tables", "allocation_benefit.json")
    json.dump(res, open(out, "w"), indent=1); print("saved", out)


if __name__ == "__main__":
    main()
