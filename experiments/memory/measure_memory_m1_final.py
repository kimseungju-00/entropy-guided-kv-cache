"""
Matched-accuracy KV cache memory of the final method vs. the baselines.

No GPU measurement: reads the accuracy-budget curves and, at each score reached by the
proposed method, finds the baseline budget reaching the same score and converts both to
KV cache memory in GB at an 8k context (memory is linear in the kept fraction).
The matching rule is defined in src/eval/matching.py and is the same for every method,
model and table/figure: isotonic fit + linear interpolation, all points kept, no
extrapolation, paired bootstrap 95% CIs when per-sample scores are available.

Run from the project root:
  python experiments/memory/measure_memory_m1_final.py [--boot 2000]
Output: results/kvpress_memory_m1/<model>_m1mem_final.json
"""
import json, os, sys, argparse
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from transformers import AutoConfig
from src.eval.matching import load_curves, matched, RULE

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

MODELS = {"Mistral-7B-Instruct-v0.3": ("Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),
          "Llama-3.1-8B-Instruct": ("LLaMA-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
          "Qwen3-8B": ("Qwen3-8B", "Qwen/Qwen3-8B")}
BASELINES = ["SnapKV", "PyramidKV", "AdaSnapKV", "CAKE", "Uniform"]
CTX = 8000
os.makedirs("results/kvpress_memory_m1", exist_ok=True)


def full_kv_gb(model_id):
    c = AutoConfig.from_pretrained(model_id)
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return 2 * c.num_hidden_layers * c.num_key_value_heads * hd * CTX * 2 / 1024**3  # K+V, bf16


for mkey, (mname, mid) in MODELS.items():
    curves = load_curves(ROOT, mkey)
    if "Proposed" not in curves:
        print(f"{mname}: no proposed-method results, skipping"); continue
    fg = full_kv_gb(mid)
    prop = curves["Proposed"]
    out = {"ctx": CTX, "full_kv_gb": fg, "rule": RULE, "baselines": {}}
    print(f"\n{'=' * 72}\n{mname} (FullKV KV {fg:.3f} GB at {CTX} tokens)\n{'=' * 72}")
    for base in BASELINES:
        if base not in curves: continue
        r = matched(prop, curves[base], boot=args.boot)
        for p in r["points"]:
            p["prop_gb"] = p["budget"] * fg
            p["base_gb"] = None if p["base_budget"] is None else p["base_budget"] * fg
        out["baselines"][base] = r
        tag = "paired bootstrap" if r["paired_bootstrap"] else "means only, no CI"
        print(f"-- vs {base} ({len(curves[base])} points, {tag})")
        for p in r["points"]:
            if p["save_pct"] is None:
                print(f"   {p['cfg']:18s} score {p['score']:.4f} budget {p['budget']:.3f}: not matched"); continue
            ci = f"  CI [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}]" if "ci" in p else ""
            print(f"   {p['cfg']:18s} score {p['score']:.4f} budget {p['budget']:.3f} vs {p['base_budget']:.3f}"
                  f"  ({p['prop_gb']:.3f} vs {p['base_gb']:.3f} GB)  {p['save_pct']:+.1f}%{ci}")
        if "mean_save_pct" in r:
            ci = f"  CI [{r['mean_ci'][0]:+.1f}, {r['mean_ci'][1]:+.1f}]" if "mean_ci" in r else ""
            print(f"   mean over {r['n_matched']} matched points: {r['mean_save_pct']:+.1f}%{ci}")
    # legacy keys (SnapKV) read by the table/figure scripts
    if "SnapKV" in out["baselines"]:
        out["points"] = [{"score": p["score"], "m1_budget": p["budget"], "snap_budget": p["base_budget"],
                          "m1_gb": p["prop_gb"], "snap_gb": p["base_gb"], "save_pct": p["save_pct"],
                          "ci": p.get("ci")}
                         for p in out["baselines"]["SnapKV"]["points"] if p["save_pct"] is not None]
    json.dump(out, open(f"results/kvpress_memory_m1/{mname}_m1mem_final.json", "w"), indent=2)
print("\nSaved: results/kvpress_memory_m1/*_m1mem_final.json")
