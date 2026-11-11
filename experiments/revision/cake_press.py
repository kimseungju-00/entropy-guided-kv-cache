"""
CAKE (Qin et al., ICLR 2025) re-implemented as a kvpress press, for comparison.

Follows the official implementation (github.com/antgroup/cakekv, cake/model/modify_llama.py
and cake/cake_cache.py) for the prefill stage:
  - observation window = last w queries; attention over all keys with a causal mask in the window
  - layer preference  P_l = H_l^(1/tau1) * V_l^(1/tau2), where H_l is the entropy summed over
    heads and window queries of the attention to non-window keys, and V_l is the variance of
    that attention across window queries, summed over heads and keys
  - eviction indicator = mean + gamma * var of window attention, avg-pooled (kernel 5),
    averaged over the query heads of each KV group; window keys are always kept
  - total non-window budget (rho*T - w) * L is split across layers proportionally to P_l
    (with the official clipping/redistribution), and every layer is evicted once all layer
    preferences are known.

Differences from the official code, applied identically to all methods in our evaluation:
  - the budget is a ratio rho of the prompt length instead of a fixed number of tokens
  - no decoding-time eviction (the cache grows with generated tokens, as for SnapKV and ours)
  - eviction after the last layer instead of cascading; cascading reduces peak prefill
    memory but yields the same final cache, since budgets only shrink as layers are added
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress  # noqa: F401  (keeps import parity)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.methods.proposed_kvpress import _window_attention

# tau values from the official config (model2tau.json, cache size 1024). Qwen3 is not in the
# official config; the Qwen2.5-7B values are used.
CAKE_TAU = {
    "Mistral-7B-Instruct-v0.3": (0.8, 0.5),
    "Llama-3.1-8B-Instruct": (1.6, 0.4),
    "Qwen3-8B": (1.5, 0.6),
}


def adjust_budgets(budgets, total, seq_len):
    """Clip per-layer budgets to seq_len and redistribute so that they sum to `total`
    (same procedure as the official implementation)."""
    b = np.array(budgets, dtype=int)
    excess = np.maximum(b - seq_len, 0); b = np.minimum(b, seq_len)
    tot_ex = int(excess.sum())
    if tot_ex > 0:
        ok = b < seq_len; n = int(ok.sum())
        if n > 0:
            b[ok] += tot_ex // n; b[np.where(ok)[0][: tot_ex % n]] += 1
    diff = int(total - b.sum())
    if diff != 0:
        ok = b < seq_len if diff > 0 else b > 1; n = int(ok.sum())
        if n > 0:
            step, rem = abs(diff) // n, abs(diff) % n
            sgn = 1 if diff > 0 else -1
            b[ok] += sgn * step; b[np.where(ok)[0][:rem]] += sgn
    return np.minimum(b, seq_len).tolist()


@dataclass
class CAKEPress(BasePress):
    keep_ratio: float = 0.2
    window_size: int = 32
    kernel_size: int = 5
    tau1: float = 1.0
    tau2: float = 1.0
    gamma: float = 200.0
    _pref: dict = field(default_factory=dict)
    _score: dict = field(default_factory=dict)

    def _layer_pref(self, attn, past):
        """Layer preference P_l = H_l^(1/tau1) * V_l^(1/tau2) from the window attention."""
        H = -(past * torch.log(past + 1e-10)).sum()
        V = past.var(dim=-2).sum()
        return float(H ** (1 / self.tau1) * V ** (1 / self.tau2))

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer, L = module.layer_idx, module.config.num_hidden_layers
        bsz, n_kv, k_len, _ = keys.shape
        w = self.window_size
        if layer == 0:
            self._pref, self._score = {}, {}
        if k_len <= w or self.keep_ratio * k_len <= w:
            return keys, values

        attn = _window_attention(module, hidden_states, keys, w, kwargs["position_embeddings"])
        attn = attn.to(hidden_states.dtype).float()        # official code casts to model dtype
        past = attn[..., :-w]                               # (B, H, w, k_len - w)
        self._pref[layer] = self._layer_pref(attn, past)

        s = past.mean(dim=-2) + self.gamma * past.var(dim=-2)          # (B, H, k_len - w)
        s = F.avg_pool1d(s, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        self._score[layer] = s.view(bsz, n_kv, -1, s.shape[-1]).mean(2)  # (B, n_kv, k_len - w)

        if layer != L - 1 or len(self._pref) != L:
            return keys, values

        # all layer preferences known: allocate and evict every layer
        total = int(round((self.keep_ratio * k_len - w) * L))
        pref = np.array([self._pref[i] for i in range(L)])
        budgets = adjust_budgets(pref / pref.sum() * total, total, k_len - w)
        cache = kwargs["past_key_values"]
        out = None
        for i, bud in enumerate(budgets):
            if i == layer:
                k, v = keys, values
            else:
                k, v = cache.layers[i].keys, cache.layers[i].values
            idx = self._score[i].topk(max(int(bud), 1), dim=-1).indices.sort(dim=-1).values
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            k_c = torch.cat([k[:, :, :-w].gather(2, idx), k[:, :, -w:]], dim=2)
            v_c = torch.cat([v[:, :, :-w].gather(2, idx), v[:, :, -w:]], dim=2)
            if hasattr(self, "_budget_log"):
                self._budget_log.append(k_c.shape[2] / k_len)
            if i == layer:
                out = (k_c, v_c)
            else:
                cache.layers[i].keys, cache.layers[i].values = k_c, v_c
        self._pref, self._score = {}, {}
        return out
