"""
Analyze the revision experiments with the shared matching rule (src/eval/matching.py).

Comparisons (each: budget saving of A vs B at matched accuracy, and score difference of A vs B
at matched budget; paired bootstrap 95% CIs):
  Proposed       vs Uniform      - effect of the entropy-based layer allocation (same press)
  Proposed       vs CAKE         - vs the closest layer-adaptive method
  Proposed-logT  vs Uniform      - absolute-entropy (log T) normalization variant
  Proposed-logT  vs Proposed
  Proposed+AdaKV vs AdaSnapKV    - layer budget + Ada-KV head allocation
  Proposed+AdaKV vs Proposed
  CAKE gap decomposition (layer allocation x token score):
    Proposed vs CAKEalloc+meanscore, Proposed+CAKEscore vs CAKE   - allocation effect
    Proposed+CAKEscore vs Proposed, CAKE vs CAKEalloc+meanscore   - token-score effect
  EntropyNorm(-a4) vs Proposed / Uniform / CAKE(alloc) - same entropy signal, fixed total budget
  Uniform / SnapKV with window 32 or 64                 - what in the press beats SnapKV
Also prints each curve and the per-layer keep-ratio range of the proposed method.

Usage: python experiments/revision/analyze_phase1.py [--boot 2000]
"""
import json, os, sys, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.eval.matching import load_curves, matched, matched_score, TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
MODELS = ["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "Qwen3-8B"]
PAIRS = [("Proposed", "Uniform"), ("Proposed", "CAKE"), ("Proposed-logT", "Uniform"),
         ("Proposed-logT", "Proposed"), ("Proposed+AdaKV", "AdaSnapKV"), ("Proposed+AdaKV", "Proposed"),
         # CAKE gap decomposition: allocation effect with each token score, score effect with each allocation
         ("Proposed", "CAKEalloc+meanscore"), ("Proposed+CAKEscore", "CAKE"),
         ("Proposed+CAKEscore", "Proposed"), ("CAKE", "CAKEalloc+meanscore"),
         ("CAKEalloc+meanscore", "Uniform"),
         # fixed-total variant (entropy preference, fixed total budget)
         ("EntropyNorm", "Proposed"), ("EntropyNorm", "Uniform"), ("EntropyNorm", "CAKEalloc+meanscore"),
         ("EntropyNorm", "CAKE"), ("EntropyNorm-a4", "Proposed"), ("EntropyNorm-a4", "CAKEalloc+meanscore"),
         # press decomposition: window length and implementation
         ("Uniform", "SnapKV"), ("SnapKV-w32", "SnapKV"), ("Uniform", "SnapKV-w32"),
         ("Uniform-w64", "SnapKV"), ("Uniform", "Uniform-w64"),
         # the other baselines with the observation window of the proposed press
         ("Proposed", "PyramidKV-w32"), ("Proposed", "AdaSnapKV-w32"), ("PyramidKV-w32", "PyramidKV"),
         ("AdaSnapKV-w32", "AdaSnapKV"), ("Proposed+AdaKV", "AdaSnapKV-w32")]
REVISION_ONLY = ["Proposed", "Uniform", "CAKE", "Proposed-logT", "Proposed+AdaKV", "AdaSnapKV",
                 "SnapKV", "PyramidKV", "Proposed+CAKEscore", "CAKEalloc+meanscore",
                 "EntropyNorm", "EntropyNorm-a4", "SnapKV-w32", "Uniform-w64", "Uniform-w16", "Uniform-w128",
                 "PyramidKV-w32", "AdaSnapKV-w32"]

for mkey in MODELS:
    curves = load_curves(ROOT, mkey, REVISION_ONLY)
    curves = {m: c for m, c in curves.items() if c[0].scores is not None}  # per-sample results only
    if not curves: continue
    print(f"\n{'=' * 72}\n{mkey}\n{'=' * 72}")
    for m, c in curves.items():
        print(f"  {m:15s}", "  ".join(f"({p.budget:.3f}, {p.score:.4f})" for p in c))
    for a, b in PAIRS:
        if a not in curves or b not in curves or len(curves[b]) < 2: continue
        r = matched(curves[a], curves[b], boot=args.boot)
        print(f"-- {a} vs {b}: budget saving at matched accuracy"
              + (" (paired bootstrap)" if r["paired_bootstrap"] else ""))
        for p in r["points"]:
            if p["save_pct"] is None:
                print(f"   {p['cfg']:14s} ({p['budget']:.3f}, {p['score']:.4f}): not matched"); continue
            ci = f"  CI [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}]" if "ci" in p else ""
            print(f"   {p['cfg']:14s} ({p['budget']:.3f}, {p['score']:.4f}) vs budget {p['base_budget']:.3f}:"
                  f" {p['save_pct']:+.1f}%{ci}")
        if "mean_save_pct" in r:
            ci = f"  CI [{r['mean_ci'][0]:+.1f}, {r['mean_ci'][1]:+.1f}]" if "mean_ci" in r else ""
            print(f"   mean over {r['n_matched']} matched points: {r['mean_save_pct']:+.1f}%{ci}")
        r = matched_score(curves[a], curves[b], boot=args.boot)
        print(f"   score difference at matched budget (points):")
        for p in r["points"]:
            if p["diff_pts"] is None:
                print(f"   {p['cfg']:14s} budget {p['budget']:.3f}: not matched"); continue
            ci = f"  CI [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]" if "ci" in p else ""
            print(f"   {p['cfg']:14s} budget {p['budget']:.3f}: {p['score']:.4f} vs {p['base_score']:.4f}"
                  f"  {p['diff_pts']:+.2f}{ci}")
        if "mean_diff_pts" in r:
            ci = f"  CI [{r['mean_ci'][0]:+.2f}, {r['mean_ci'][1]:+.2f}]" if "mean_ci" in r else ""
            print(f"   mean over {r['n_matched']} matched points: {r['mean_diff_pts']:+.2f}{ci}")
    # per-layer keep ratio range of the proposed method
    path = os.path.join(ROOT, "results", "revision_phase1", f"{mkey}.json")
    if os.path.exists(path) and "Proposed" in curves:
        res = json.load(open(path))
        for p in curves["Proposed"]:
            lk = [res[p.cfg][t]["layer_keep_mean"] for t in TASKS if res[p.cfg][t].get("layer_keep_mean")]
            if lk:
                lk = np.mean(lk, axis=0)
                print(f"   {p.cfg}: per-layer keep ratio {lk.min():.3f} (L{lk.argmin()}) - {lk.max():.3f} (L{lk.argmax()})")
