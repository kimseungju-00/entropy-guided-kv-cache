"""
Relation of the layer statistic phi to the per-layer discarded mass (measure_layer_delta.py).

Reports for each model:
  - across-layer correlation of the mean profiles: phi vs delta(rho), mean entropy H vs delta(rho),
    H / log T vs delta(rho), phi vs H (Pearson and Spearman; bootstrap 95% CI over samples)
  - mean within-sample correlation of phi and delta across layers
  - total discarded mass under the proposed rule versus a uniform ratio with the same mean ratio
    in the same sample, at each epsilon (paired bootstrap 95% CI)
Layers listed in EXCLUDE are also left out in a second pass, to check the influence of the
near-uniform layers of Qwen3-8B.

Usage: python experiments/revision/analyze_layer_delta.py [--boot 10000]
Output: results/tables/layer_delta.json
"""
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODELS = ["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "Qwen3-8B"]
TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
EPS = [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
RMIN, RMAX = 0.05, 0.50
EXCLUDE = {"Qwen3-8B": [21, 23, 24]}


def load(model):
    d = json.load(open(os.path.join(ROOT, "results", "revision_phase1", f"{model}_layerdelta.json")))
    rhos = np.array(d["_meta"]["rhos"]); per = []
    for t in TASKS:
        rows = d[t]
        per.append(dict(phi=np.array([r["phi"] for r in rows]), H=np.array([r["H"] for r in rows]),
                        logT=np.array([r["logT"] for r in rows]), delta=np.array([r["delta"] for r in rows])))
    return rhos, per


def corr_ci(a, b):
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b)[0])


def resample(per, rng, boot):
    for _ in range(boot):
        yield [rng.integers(0, len(p["phi"]), len(p["phi"])) for p in per]


def profiles(per, idx=None):
    out = {}
    for key in ("phi", "H", "delta"):
        out[key] = np.mean([(p[key] if idx is None else p[key][i]).mean(0) for p, i in
                            zip(per, idx or [None] * len(per))], axis=0)
    out["Hn"] = np.mean([((p["H"] / p["logT"][:, None]) if idx is None else (p["H"] / p["logT"][:, None])[i]).mean(0)
                         for p, i in zip(per, idx or [None] * len(per))], axis=0)
    return out


def interp_delta(rhos, delta, rho):
    """delta(rho) for (S, L, R) grid values and (S, L) ratios, linear in rho."""
    S, L, _ = delta.shape; out = np.empty((S, L))
    i = np.clip(np.searchsorted(rhos, rho) - 1, 0, len(rhos) - 2); w = (rho - rhos[i]) / (rhos[i + 1] - rhos[i])
    for s in range(S):
        l = np.arange(L); out[s] = delta[s, l, i[s]] * (1 - w[s]) + delta[s, l, i[s] + 1] * w[s]
    return out


def alloc(rhos, per, eps):
    """Per-sample mean delta under the rule and under the uniform ratio with the same mean, per task."""
    res = []
    for p in per:
        r = np.clip(RMIN + (RMAX - RMIN) * p["phi"] ** (1 / eps), RMIN, RMAX)
        u = np.repeat(r.mean(1, keepdims=True), r.shape[1], axis=1)
        res.append((interp_delta(rhos, p["delta"], r).mean(1), interp_delta(rhos, p["delta"], u).mean(1), r.mean()))
    return res


def analyze(model, boot, keep=None):
    rhos, per = load(model)
    if keep is not None:
        per = [dict(phi=p["phi"][:, keep], H=p["H"][:, keep], logT=p["logT"], delta=p["delta"][:, keep]) for p in per]
    rng = np.random.default_rng(0); out = {"n_samples": int(sum(len(p["phi"]) for p in per))}
    base = profiles(per); boots = [profiles(per, idx) for idx in resample(per, rng, boot)]
    pairs = {"phi_H": ("phi", "H")}
    for j, r in enumerate(rhos):
        for name, key in (("phi", "phi"), ("H", "H"), ("Hn", "Hn")):
            pairs[f"{name}_delta_{r}"] = (key, ("delta", j))
    get = lambda pr, k: pr[k] if isinstance(k, str) else pr[k[0]][:, k[1]]
    for name, (a, b) in pairs.items():
        pe, sp = corr_ci(get(base, a), get(base, b))
        bs = [np.corrcoef(get(x, a), get(x, b))[0, 1] for x in boots]
        out[name] = {"pearson": pe, "spearman": sp, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    j = list(rhos).index(0.2)
    within = [np.corrcoef(p["phi"][s], p["delta"][s, :, j])[0, 1] for p in per for s in range(len(p["phi"]))]
    out["within_sample_phi_delta_0.2"] = {"mean": float(np.nanmean(within)), "sd": float(np.nanstd(within))}
    if keep is None:
        tab = {}
        for eps in EPS:
            a = alloc(rhos, per, eps)
            est = lambda idx: float(np.mean([(x[0][i] - x[1][i]).mean() for x, i in zip(a, idx)]))
            all_idx = [np.arange(len(x[0])) for x in a]
            bs = [est(idx) for idx in resample(per, rng, boot)]
            tab[str(eps)] = {"mean_ratio": float(np.mean([x[2] for x in a])),
                             "delta_rule": float(np.mean([x[0].mean() for x in a])),
                             "delta_uniform": float(np.mean([x[1].mean() for x in a])),
                             "diff": est(all_idx), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        out["rule_vs_uniform"] = tab
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(); res = {}
    for m in MODELS:
        p = os.path.join(ROOT, "results", "revision_phase1", f"{m}_layerdelta.json")
        if not os.path.exists(p) or any(t not in json.load(open(p)) for t in TASKS): continue
        res[m] = analyze(m, a.boot)
        if m in EXCLUDE:
            n = len(load(m)[1][0]["phi"][0]); keep = [l for l in range(n) if l not in EXCLUDE[m]]
            res[m]["without_near_uniform_layers"] = analyze(m, a.boot, keep)
    out = os.path.join(ROOT, "results", "tables", "layer_delta.json")
    json.dump(res, open(out, "w"), indent=1); print("saved", out)
    for m, r in res.items():
        print(m, r["n_samples"], "phi_H", {k: round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] for k, v in r["phi_H"].items()})
        for rho in (0.05, 0.2, 0.5):
            print(f"  rho={rho}", *[f"{n}: {r[f'{n}_delta_{rho}']['pearson']:+.3f} sp {r[f'{n}_delta_{rho}']['spearman']:+.3f} {np.round(r[f'{n}_delta_{rho}']['ci'], 3)}" for n in ("phi", "H", "Hn")])
        print("  within", r["within_sample_phi_delta_0.2"])
        for e, t in r["rule_vs_uniform"].items():
            print(f"  eps {e:>4} ratio {t['mean_ratio']:.3f} rule {t['delta_rule']:.4f} uniform {t['delta_uniform']:.4f} diff {t['diff']:+.4f} {np.round(t['ci'], 4)}")
        if "without_near_uniform_layers" in r:
            w = r["without_near_uniform_layers"]
            print("  w/o near-uniform: phi_H", round(w["phi_H"]["pearson"], 3), round(w["phi_H"]["spearman"], 3), "phi_delta_0.2", round(w["phi_delta_0.2"]["pearson"], 3), round(w["phi_delta_0.2"]["spearman"], 3))


if __name__ == "__main__":
    main()
