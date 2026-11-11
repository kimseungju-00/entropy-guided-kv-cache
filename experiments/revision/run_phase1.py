"""
Revision phase 1: layer-adaptive budget (EntropyBudgetPress) vs the same press
with a uniform keep ratio, over a wide budget range, saving per-sample scores.

Both methods share token selection, window size and kernel; only the per-layer
keep ratio differs. This isolates the effect of the entropy-based allocation.

Saved per (config, task):
  - per-sample scores (for paired bootstrap CIs)
  - per-sample realized budget (mean keep ratio over layers)
  - for M1: per-layer keep ratio, phi, mean/max head entropy (averaged over samples)

Results are written after every (config, task), so the run can be resumed.

Usage:
  quick check : CUDA_VISIBLE_DEVICES=0 python experiments/revision/run_phase1.py --model <m> --n 2 --tasks hotpotqa --configs M1_e0.5 Uni_0.2 --tag test
  full run    : CUDA_VISIBLE_DEVICES=0 python experiments/revision/run_phase1.py --model <m>
Output: results/revision_phase1/<model>[_<tag>].json
"""
import sys, os, argparse, json, time, gc, math
from dataclasses import dataclass
import numpy as np
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger("kvpress").setLevel(logging.ERROR)
from transformers import pipeline
import kvpress
from src.methods.proposed_kvpress import EntropyBudgetPress
sys.path.insert(0, os.path.join(ROOT, "experiments", "revision"))
from cake_press import CAKEPress, CAKE_TAU
from combo_press import EntropyScorer
from src.eval.longbench_loader import load_task
from src.eval.evaluator import score_sample, TASK_MAX_NEW, DATASET2PROMPT

M1_EPS = [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
UNI_KEEP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
# interleave so that a partial run already covers both curves
DEFAULT_CONFIGS = [c for pair in zip(
    [f"M1_e{e}" for e in [0.5, 0.3, 1.0, 0.2, 2.0, 0.7]],
    [f"Uni_{k}" for k in [0.2, 0.1, 0.3, 0.15, 0.4, 0.25]]) for c in pair] + \
    [f"M1_e{e}" for e in [0.4, 0.15, 1.5, 3.0]]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--tasks", nargs="+", default=["qasper", "hotpotqa", "gov_report", "trec", "lcc"])
ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
ap.add_argument("--max_seq", type=int, default=10000)
ap.add_argument("--outdir", default="results/revision_phase1")
ap.add_argument("--tag", default="")
args = ap.parse_args()
RAW_TASKS = {"lcc", "repobench-p"}
mname = args.model.split("/")[-1]


@dataclass
class M1Logged(EntropyBudgetPress):
    """EntropyBudgetPress that also records phi and absolute entropy per layer."""
    def _layer_keep_ratio(self, entropy):
        keep = super()._layer_keep_ratio(entropy)
        h_norm = entropy / (entropy.max() + 1e-9)
        self._stat_log.append((h_norm.mean().item(), entropy.mean().item(), entropy.max().item()))
        return keep


@dataclass
class AbsEntropyPress(M1Logged):
    """Variant: phi from absolute entropy, each head entropy divided by log T (T = number of
    keys, the maximum possible entropy) instead of the layer maximum. Not scale invariant."""
    def _layer_keep_ratio(self, entropy):
        T = self._k_len
        phi = float((entropy / math.log(max(T, 2))).clamp(0, 1).mean())
        self._stat_log.append((phi, entropy.mean().item(), entropy.max().item()))
        g = phi ** (1.0 / max(self.epsilon, 0.01))
        return float(np.clip(self.min_ratio + (self.max_ratio - self.min_ratio) * g, self.min_ratio, self.max_ratio))

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        self._k_len = keys.shape[2]
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)


@dataclass
class M1VarPress(M1Logged):
    """Decomposition: proposed layer allocation + CAKE token score (mean + gamma * var of
    the window attention, computed in the model dtype as in the official CAKE code)."""
    gamma: float = 200.0
    def _token_scores(self, attn):
        a = attn.to(torch.bfloat16).float()
        return a.mean(dim=-2) + self.gamma * a.var(dim=-2)


@dataclass
class EntNormPress(CAKEPress):
    """Budget-normalized variant: the proposed layer signal phi (mean of H / max H over heads,
    from the window-averaged attention) gives the preference phi^alpha, and a fixed total budget
    rho is split across layers in proportion to it (CAKE's allocation step). The layer contrast
    therefore no longer shrinks as the budget grows. Mean token score (gamma = 0)."""
    alpha: float = 2.0
    gamma: float = 0.0
    def _layer_pref(self, attn, past):
        head_attn = attn.mean(dim=-2)[0]
        p = head_attn / (head_attn.sum(dim=-1, keepdim=True) + 1e-9)
        h = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(dim=-1)
        phi = (h / (h.max() + 1e-9)).mean().item()
        self._stat_log.append((phi, h.mean().item(), h.max().item()))
        return phi ** self.alpha


@dataclass
class UniformPress(EntropyBudgetPress):
    """Same press, same keep ratio for every layer."""
    fixed_keep: float = 0.2
    def _layer_keep_ratio(self, entropy):
        return float(self.fixed_keep)


def make_press(cfg):
    if cfg.startswith("M1_e"):
        p = M1Logged(epsilon=float(cfg[4:]))
    elif cfg.startswith("M1Ada_e"):
        p = kvpress.AdaKVPress(EntropyScorer(epsilon=float(cfg[7:])))
    elif cfg.startswith("AdaSnap_"):
        k = float(cfg[8:])
        p = kvpress.AdaKVPress(kvpress.SnapKVPress(compression_ratio=round(1.0 - k, 4)))
        p._fixed_keep = k
    elif cfg.startswith("Snap32_"):  # press decomposition: SnapKV with the observation window of ours
        k = float(cfg[7:])
        p = kvpress.SnapKVPress(compression_ratio=round(1.0 - k, 4), window_size=32)
        p._fixed_keep = k
    elif cfg.startswith("Pyr32_"):  # PyramidKV with the observation window of the proposed press
        k = float(cfg[6:])
        p = kvpress.PyramidKVPress(compression_ratio=round(1.0 - k, 4), window_size=32)
        p._fixed_keep = k
    elif cfg.startswith("Ada32_"):  # Ada-SnapKV with the observation window of the proposed press
        k = float(cfg[6:])
        p = kvpress.AdaKVPress(kvpress.SnapKVPress(compression_ratio=round(1.0 - k, 4), window_size=32))
        p._fixed_keep = k
    elif cfg.startswith("UniW"):  # press decomposition: uniform ratio with another window, UniW{w}_{keep}
        w, k = cfg[4:].split("_")
        p = UniformPress(fixed_keep=float(k), window_size=int(w))
    elif cfg.startswith(("Snap_", "Pyr_")):
        k = float(cfg.split("_")[1])
        cls = kvpress.SnapKVPress if cfg.startswith("Snap_") else kvpress.PyramidKVPress
        p = cls(compression_ratio=round(1.0 - k, 4))
        p._fixed_keep = k
    elif cfg.startswith("Abs_e"):
        p = AbsEntropyPress(epsilon=float(cfg[5:]))
    elif cfg.startswith("Uni_"):
        p = UniformPress(fixed_keep=float(cfg[4:]))
    elif cfg.startswith("M1Var_e"):
        p = M1VarPress(epsilon=float(cfg[7:]))
    elif cfg.startswith("EntNormA4_"):
        p = EntNormPress(keep_ratio=float(cfg[10:]), alpha=4.0)
    elif cfg.startswith("EntNorm_"):  # alpha = 2 = 1 / eps at the default eps = 0.5
        p = EntNormPress(keep_ratio=float(cfg[8:]), alpha=2.0)
    elif cfg.startswith("CAKEMean_"):  # decomposition: CAKE layer allocation + mean token score
        t1, t2 = CAKE_TAU[mname]
        p = CAKEPress(keep_ratio=float(cfg[9:]), tau1=t1, tau2=t2, gamma=0.0)
    elif cfg.startswith("CAKE_"):
        t1, t2 = CAKE_TAU[mname]
        p = CAKEPress(keep_ratio=float(cfg[5:]), tau1=t1, tau2=t2)
    else:
        raise ValueError(cfg)
    p._budget_log, p._stat_log = [], []
    if hasattr(p, "press"):  # AdaKV wrapper: the inner scorer logs into the same lists
        p.press._budget_log, p.press._stat_log = p._budget_log, p._stat_log
    return p


def split(s, task):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    return pre + s["context"], post.replace("{input}", s.get("input", ""))
def truncate(tok, ctx, q, ms):
    qi = tok(q, add_special_tokens=False)["input_ids"]; b = ms - len(qi) - 64
    ci = tok(ctx, add_special_tokens=False)["input_ids"]
    if len(ci) > b:
        h = b//2; ci = ci[:h]+ci[-h:]; ctx = tok.decode(ci, skip_special_tokens=True)
    return ctx, q

print(f"Loading {args.model}  [GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}]")
pipe = pipeline("kv-press-text-generation", model=args.model, device="cuda",
                torch_dtype=torch.bfloat16, model_kwargs={"attn_implementation": "flash_attention_2"})
tok = pipe.tokenizer
saved = tok.chat_template
if tok.bos_token is None: tok.bos_token = ""
n_layers = pipe.model.config.num_hidden_layers
print(f"Loaded ({n_layers} layers)\n")

def set_tmpl(task): tok.chat_template = None if task in RAW_TASKS else saved
def fin(ans, task): return ans if task in RAW_TASKS else ans.strip()


def run_task(task, cfg):
    """Same evaluation as run_performance.py, plus per-sample and per-layer logs."""
    set_tmpl(task); mn = TASK_MAX_NEW.get(task, 32)
    press = make_press(cfg)
    ids, scores, budgets = [], [], []
    layer_keep, layer_stat = [], []
    for i, s in enumerate(load_task(task, max_samples=args.n)):
        ctx, q = split(s, task); ctx, q = truncate(tok, ctx, q, args.max_seq)
        b0, s0 = len(press._budget_log), len(press._stat_log)
        try:
            ans = pipe(ctx, question=q, max_new_tokens=mn, press=press)["answer"]
            a = s.get("answers", []); a = [a] if isinstance(a, str) else a
            sc = score_sample(fin(ans, task), a, task, s.get("all_classes"))
        except Exception as e:
            print(f"      sample {i} error: {type(e).__name__} {str(e)[:80]}")
            continue
        kb = press._budget_log[b0:]; st = press._stat_log[s0:]
        ids.append(s.get("_id", i)); scores.append(float(sc))
        budgets.append(float(np.mean(kb)) if kb else getattr(press, "_fixed_keep", 1.0))
        if len(kb) == n_layers: layer_keep.append(kb)
        if len(st) == n_layers: layer_stat.append(st)
    out = {"n": len(scores), "score": float(np.mean(scores)) if scores else 0.0,
           "budget": float(np.mean(budgets)) if budgets else None,
           "ids": ids, "scores": scores, "budgets": budgets}
    if layer_keep:
        out["layer_keep_mean"] = np.mean(layer_keep, axis=0).tolist()
    if layer_stat:
        a = np.mean(layer_stat, axis=0)  # (L, 3)
        out["layer_phi_mean"] = a[:, 0].tolist()
        out["layer_H_mean"] = a[:, 1].tolist()
        out["layer_H_max"] = a[:, 2].tolist()
    torch.cuda.empty_cache()
    return out


os.makedirs(args.outdir, exist_ok=True)
outpath = os.path.join(args.outdir, f"{mname}{'_' + args.tag if args.tag else ''}.json")
results = json.load(open(outpath)) if os.path.exists(outpath) else {}
results.setdefault("_meta", {"model": args.model, "n": args.n, "max_seq": args.max_seq,
                             "window_size": 32, "kernel_size": 5, "min_ratio": 0.05, "max_ratio": 0.50})

print(f"=== {mname} | phase 1 (n={args.n}) -> {outpath} ===")
for cfg in args.configs:
    results.setdefault(cfg, {})
    print(f"[{cfg}]")
    for task in args.tasks:
        if task in results[cfg] and results[cfg][task].get("n_req") == args.n:
            print(f"    {task:12s} done, skip"); continue
        t0 = time.time()
        r = run_task(task, cfg)
        r["sec"] = round(time.time() - t0); r["n_req"] = args.n
        results[cfg][task] = r
        tmp = outpath + ".tmp"
        with open(tmp, "w") as f: json.dump(results, f)
        os.replace(tmp, outpath)
        print(f"    {task:12s} score={r['score']:.4f} budget={r['budget'] or 0:.4f} (n={r['n']}, {r['sec']}s)", flush=True)
    gc.collect(); torch.cuda.empty_cache()
print(f"\nDone: {outpath}")
