"""
Per-layer discarded attention mass versus the layer statistic phi.

For every sample and layer, this measures on the same observation-window attention the method
uses: phi (equation 6), the mean head entropy H, log T, and the discarded mass delta at a grid
of retention ratios, with the retained set chosen exactly as the method chooses it (window
attention averaged over the window, average pooling, GQA group mean, window keys always kept).
delta is averaged over the query heads of the layer. Nothing is evicted, so all layers see the
uncompressed prefill and delta at any allocation can be computed afterwards from the grid.

Usage: CUDA_VISIBLE_DEVICES=0 python experiments/revision/measure_layer_delta.py --model <m>
Output: results/revision_phase1/<model>_layerdelta.json
"""
import sys, os, argparse, json, math, time
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger("kvpress").setLevel(logging.ERROR)
from transformers import pipeline
import kvpress  # noqa: F401  (registers the pipeline)
from src.methods.proposed_kvpress import EntropyBudgetPress, _window_attention
from src.eval.longbench_loader import load_task
from src.eval.evaluator import DATASET2PROMPT

RHOS = [0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--tasks", nargs="+", default=["qasper", "hotpotqa", "gov_report", "trec", "lcc"])
ap.add_argument("--max_seq", type=int, default=10000)
ap.add_argument("--outdir", default="results/revision_phase1")
args = ap.parse_args()
RAW_TASKS = {"lcc", "repobench-p"}
mname = args.model.split("/")[-1]


@dataclass
class DeltaLogger(EntropyBudgetPress):
    """Records the per-layer statistics and leaves the cache untouched."""
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, n_kv, k_len, _ = keys.shape
        n_group = module.config.num_attention_heads // n_kv
        w = self.window_size
        if hidden_states.shape[1] <= w:
            return keys, values
        attn = _window_attention(module, hidden_states, keys, w, kwargs["position_embeddings"]).float()
        head_attn = attn.mean(dim=-2)[0]
        p = head_attn / (head_attn.sum(dim=-1, keepdim=True) + 1e-9)  # (num_heads, k_len)
        logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
        h = -(p * logp).sum(dim=-1)
        phi = (h / (h.max() + 1e-9)).mean().item()
        scores = F.avg_pool1d(attn.mean(dim=-2), kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2, stride=1)
        scores = scores.view(bsz, n_kv, n_group, k_len).mean(2)
        scores[..., -w:] = scores.max().item() + 1
        order = scores[0].argsort(dim=-1, descending=True)  # (n_kv, k_len), the method's ranking
        grouped = p.view(n_kv, n_group, k_len)
        ranked = grouped.gather(2, order.unsqueeze(1).expand(-1, n_group, -1))
        kept = ranked.cumsum(dim=-1)  # retained mass as a function of the number of kept keys
        delta = [float(1.0 - kept[..., max(1, int(k_len * r)) - 1].mean()) for r in RHOS]
        self._log.append({"phi": phi, "H": h.mean().item(), "logT": math.log(max(k_len, 2)), "delta": delta})
        return keys, values


def split(s, task):
    tmpl = DATASET2PROMPT[task]; pre, post = tmpl.split("{context}", 1)
    return pre + s["context"], post.replace("{input}", s.get("input", ""))
def truncate(tok, ctx, q, ms):
    qi = tok(q, add_special_tokens=False)["input_ids"]; b = ms - len(qi) - 64
    ci = tok(ctx, add_special_tokens=False)["input_ids"]
    if len(ci) > b:
        h = b // 2; ci = ci[:h] + ci[-h:]; ctx = tok.decode(ci, skip_special_tokens=True)
    return ctx, q


print(f"Loading {args.model}  [GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}]")
pipe = pipeline("kv-press-text-generation", model=args.model, device="cuda",
                torch_dtype=torch.bfloat16, model_kwargs={"attn_implementation": "flash_attention_2"})
tok = pipe.tokenizer
saved = tok.chat_template
if tok.bos_token is None: tok.bos_token = ""
n_layers = pipe.model.config.num_hidden_layers
press = DeltaLogger(); press._log = []
os.makedirs(args.outdir, exist_ok=True)
outpath = os.path.join(args.outdir, f"{mname}_layerdelta.json")
results = {"_meta": {"model": args.model, "n": args.n, "rhos": RHOS, "window_size": press.window_size,
                     "kernel_size": press.kernel_size}}
for task in args.tasks:
    tok.chat_template = None if task in RAW_TASKS else saved
    t0 = time.time(); rows = []
    for i, s in enumerate(load_task(task, max_samples=args.n)):
        ctx, q = split(s, task); ctx, q = truncate(tok, ctx, q, args.max_seq)
        press._log.clear()
        pipe(ctx, question=q, max_new_tokens=1, press=press)
        if len(press._log) == n_layers:
            rows.append({"id": s.get("_id", i), "phi": [round(x["phi"], 5) for x in press._log],
                         "H": [round(x["H"], 5) for x in press._log], "logT": round(press._log[0]["logT"], 5),
                         "delta": [[round(d, 5) for d in x["delta"]] for x in press._log]})
        torch.cuda.empty_cache()
    results[task] = rows
    with open(outpath + ".tmp", "w") as f: json.dump(results, f)
    os.replace(outpath + ".tmp", outpath)
    print(f"  {task}: {len(rows)} samples, {time.time() - t0:.0f}s", flush=True)
print("Done:", outpath)
