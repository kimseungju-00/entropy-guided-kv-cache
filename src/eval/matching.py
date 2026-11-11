"""
Matched-accuracy budget comparison, shared by all memory tables, figures and analyses.

Rule (applied identically to every method and model):
  1. Each method's accuracy-budget curve uses all measured points (mean score over the five
     tasks vs. realized budget). No point is excluded.
  2. The baseline curve is made non-decreasing by isotonic regression (pool adjacent
     violators), so that a score maps to a single budget.
  3. For each point of the compared method, the baseline budget reaching the same score is
     obtained by linear interpolation on the fitted curve. A score outside the fitted range
     is reported as "not matched" and is not extrapolated.
  4. Saving = (baseline budget - method budget) / baseline budget. KV memory is linear in
     the budget, so this is also the KV memory saving.
  5. Complementary metric (matched budget): the baseline score at the method's budget is
     read from the same fitted curve (linear interpolation, no extrapolation), and the
     difference method - baseline is reported in score points (x100).
  6. When per-sample scores are available for both methods on the same samples, a paired
     bootstrap (samples resampled within each task, identically for all configurations)
     gives 95% confidence intervals for each point and for the mean over matched points
     (both metrics).

Data sources: results/revision_phase1/*.json (per-sample) are preferred; the original
results/kvpress_performance/<model>_perf.json (means only) is the fallback.
"""
import json, os, warnings
import numpy as np

TASKS = ["qasper", "hotpotqa", "gov_report", "trec", "lcc"]
RULE = ("isotonic fit of the baseline curve + linear interpolation, all points kept, "
        "no extrapolation; paired bootstrap 95% CI over samples within tasks")

# method name -> (revision file suffix, config prefix); old perf JSON key as fallback
REVISION = {
    "Proposed": ("", "M1_e"), "Uniform": ("", "Uni_"),
    "SnapKV": ("_snap", "Snap_"), "PyramidKV": ("_pyr", "Pyr_"),
    "AdaSnapKV": ("_ada", "AdaSnap_"), "Proposed+AdaKV": ("_ada", "M1Ada_e"),
    "CAKE": ("_cake", "CAKE_"), "Proposed-logT": ("_abs", "Abs_e"),
    # decomposition of the CAKE gap: layer allocation x token score
    "Proposed+CAKEscore": ("_decomp", "M1Var_e"), "CAKEalloc+meanscore": ("_decomp", "CAKEMean_"),
    # fixed-total variant: phi^alpha preference, fixed total budget split across layers
    # press decomposition: observation window (32 vs 64) x implementation (ours vs kvpress SnapKV)
    "SnapKV-w32": ("_press", "Snap32_"), "Uniform-w64": ("_press", "UniW64_"),
    "Uniform-w16": ("_press", "UniW16_"), "Uniform-w128": ("_press", "UniW128_"),
    # baselines with the observation window of the proposed press
    "PyramidKV-w32": ("_w32", "Pyr32_"), "AdaSnapKV-w32": ("_w32", "Ada32_"),
    "EntropyNorm": ("_norm", "EntNorm_"), "EntropyNorm-a4": ("_norm", "EntNormA4_"),
}
OLD_KEY = {"Proposed": "Proposed_M1", "SnapKV": "SnapKV", "PyramidKV": "PyramidKV",
           "AdaSnapKV": "AdaSnapKV", "RoleWeighted": "Proposed"}  # RoleWeighted: earlier variant
OLD_ADAPTIVE = {"Proposed", "RoleWeighted"}  # budgets are realized per task, not a fixed keep


def isotonic(y):
    """Non-decreasing least-squares fit (PAVA, equal weights)."""
    vals, wts, idx = [], [], []
    for i, v in enumerate(y):
        vals.append(float(v)); wts.append(1); idx.append([i])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, i2 = vals.pop(), wts.pop(), idx.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2); wts[-1] += w2; idx[-1] += i2
    out = np.empty(len(y))
    for v, ii in zip(vals, idx): out[ii] = v
    return out


def budget_at(xs, fit, target):
    """Smallest budget on a non-decreasing curve reaching `target`; None outside its range."""
    if len(xs) < 2 or target < fit[0] or target > fit[-1]: return None
    for i in range(1, len(xs)):
        if fit[i - 1] <= target <= fit[i]:
            if fit[i] == fit[i - 1]: return float(xs[i - 1])
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * (target - fit[i - 1]) / (fit[i] - fit[i - 1]))
    return None


class Point:
    def __init__(self, cfg, budget, score=None, scores=None, budgets=None, ids=None):
        self.cfg, self.budget = cfg, float(budget)
        self.scores = scores      # {task: np.array per-sample} or None
        self.budgets = budgets    # {task: np.array per-sample realized budget} or None
        self.ids = ids
        self.score = float(score if score is not None else self.mean_score())

    def mean_score(self, idx=None):
        return float(np.mean([(s if idx is None else s[idx[t]]).mean() for t, s in self.scores.items()]))

    def mean_budget(self, idx=None):
        if self.budgets is None or idx is None: return self.budget
        return float(np.mean([b[idx[t]].mean() for t, b in self.budgets.items()]))


def load_curves(root, model_key, methods=None):
    """{method: [Point]} sorted by budget, from revision results or the original perf JSON."""
    out = {}
    rev_dir = os.path.join(root, "results", "revision_phase1")
    old = os.path.join(root, "results", "kvpress_performance", f"{model_key}_perf.json")
    old = json.load(open(old)) if os.path.exists(old) else {}
    for m in methods or REVISION:
        suffix, prefix = REVISION.get(m, (None, None))
        path = os.path.join(rev_dir, f"{model_key}{suffix}.json") if suffix is not None else ""
        pts = []
        if path and os.path.exists(path):
            res = json.load(open(path))
            for cfg, tasks in res.items():
                if not cfg.startswith(prefix) or any(t not in tasks for t in TASKS): continue
                pts.append(Point(cfg, np.mean([tasks[t]["budget"] for t in TASKS]),
                                 scores={t: np.array(tasks[t]["scores"]) for t in TASKS},
                                 budgets={t: np.array(tasks[t]["budgets"]) for t in TASKS},
                                 ids={t: tasks[t]["ids"] for t in TASKS}))
        if not pts and m in OLD_KEY and OLD_KEY[m] in old:
            for tag, info in old[OLD_KEY[m]].items():
                t = info["tasks"]
                if any(x not in t for x in TASKS): continue
                b = np.mean([t[x]["budget"] for x in TASKS]) if m in OLD_ADAPTIVE else info["keep"]
                pts.append(Point(tag, b, score=np.mean([t[x]["score"] for x in TASKS])))
        if pts: out[m] = sorted(pts, key=lambda p: p.budget)
    return out


def score_at(xs, fit, budget):
    """Score of a non-decreasing curve at `budget` (linear interpolation); None outside its range."""
    if len(xs) < 2 or budget < xs[0] or budget > xs[-1]: return None
    return float(np.interp(budget, xs, fit))


def _savings(prop, base, idx=None):
    xs = np.array([p.budget for p in base])
    fit = isotonic([p.mean_score(idx) if idx is not None else p.score for p in base])
    out = []
    for p in prop:
        s = p.mean_score(idx) if idx is not None else p.score
        bb = budget_at(xs, fit, s)
        pb = p.mean_budget(idx)
        out.append((np.nan, np.nan) if bb is None else (bb, (bb - pb) / bb * 100))
    return np.array(out).reshape(-1, 2)


def _gains(prop, base, idx=None):
    """(baseline score at the method's budget, method - baseline score in points) per point."""
    xs = np.array([p.budget for p in base])
    fit = isotonic([p.mean_score(idx) if idx is not None else p.score for p in base])
    out = []
    for p in prop:
        s = p.mean_score(idx) if idx is not None else p.score
        bs = score_at(xs, fit, p.mean_budget(idx))
        out.append((np.nan, np.nan) if bs is None else (bs, (s - bs) * 100))
    return np.array(out).reshape(-1, 2)


def _bootstrap(prop, base, fn, boot, seed):
    """(boot x n_prop) array of fn(...)[:, 1] over paired resamples, or None when not paired."""
    if not boot or not paired(prop, base): return None
    rng = np.random.default_rng(seed)
    n = {t: len(prop[0].scores[t]) for t in TASKS}
    return np.array([fn(prop, base, {t: rng.integers(0, n[t], n[t]) for t in TASKS})[:, 1]
                     for _ in range(boot)])


def paired(prop, base):
    """True when every point of both curves has per-sample scores on the same samples."""
    pts = prop + base
    if any(p.scores is None for p in pts): return False
    ref = pts[0].ids
    return all(p.ids == ref for p in pts)


def matched(prop, base, boot=2000, seed=0):
    """Per-point and mean budget saving of `prop` vs `base` at matched accuracy."""
    est = _savings(prop, base)
    boots = _bootstrap(prop, base, _savings, boot, seed)
    res = {"points": [], "paired_bootstrap": boots is not None}
    for i, p in enumerate(prop):
        bb, sv = est[i]
        row = {"cfg": p.cfg, "score": p.score, "budget": p.budget,
               "base_budget": None if np.isnan(bb) else float(bb),
               "save_pct": None if np.isnan(sv) else float(sv)}
        if boots is not None and not np.isnan(sv):
            col = boots[:, i]; col = col[~np.isnan(col)]
            row["matched_frac"] = len(col) / len(boots)
            if len(col) >= 0.5 * len(boots):
                row["ci"] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]
        res["points"].append(row)
    ok = ~np.isnan(est[:, 1])
    res["n_matched"] = int(ok.sum())
    if ok.any():
        res["mean_save_pct"] = float(est[ok, 1].mean())
        if boots is not None:
            with warnings.catch_warnings():  # resamples with no matched point give NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                mb = np.nanmean(boots[:, ok], axis=1)
            res["mean_ci"] = [float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5))]
    return res


def matched_score(prop, base, boot=2000, seed=0):
    """Per-point and mean score difference (points) of `prop` vs `base` at matched budget.

    The baseline score at each budget of the method comes from the same isotonic fit and
    linear interpolation; budgets outside the baseline's measured range are not matched.
    """
    est = _gains(prop, base)
    boots = _bootstrap(prop, base, _gains, boot, seed)
    res = {"points": [], "paired_bootstrap": boots is not None}
    for i, p in enumerate(prop):
        bs, d = est[i]
        row = {"cfg": p.cfg, "score": p.score, "budget": p.budget,
               "base_score": None if np.isnan(bs) else float(bs),
               "diff_pts": None if np.isnan(d) else float(d)}
        if boots is not None and not np.isnan(d):
            col = boots[:, i]; col = col[~np.isnan(col)]
            row["matched_frac"] = len(col) / len(boots)
            if len(col) >= 0.5 * len(boots):
                row["ci"] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]
        res["points"].append(row)
    ok = ~np.isnan(est[:, 1])
    res["n_matched"] = int(ok.sum())
    if ok.any():
        res["mean_diff_pts"] = float(est[ok, 1].mean())
        if boots is not None:
            with warnings.catch_warnings():  # resamples with no matched point give NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                mb = np.nanmean(boots[:, ok], axis=1)
            res["mean_ci"] = [float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5))]
    return res
