"""
Combination of the proposed layer-wise budget with Ada-KV head-wise allocation.

EntropyScorer is a ScorerPress that scores tokens exactly as EntropyBudgetPress does
(window-averaged attention, avg pooling, GQA-group mean, observation window always kept)
and, for each layer, sets its compression ratio from the entropy-based keep ratio of that
layer. Wrapped in kvpress.AdaKVPress, the layer budget is then distributed across KV heads
by Ada-KV (top-k across heads with a per-head safeguard).

Usage: AdaKVPress(EntropyScorer(epsilon=0.5))
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.methods.proposed_kvpress import EntropyBudgetPress, _window_attention


@dataclass
class EntropyScorer(ScorerPress):
    compression_ratio: float = 0.5  # overwritten per layer in score(); must be > 0 initially
    epsilon: float = 0.5
    window_size: int = 32
    kernel_size: int = 5
    min_ratio: float = 0.05
    max_ratio: float = 0.50

    def __post_init__(self):
        self._rule = EntropyBudgetPress(epsilon=self.epsilon, min_ratio=self.min_ratio,
                                        max_ratio=self.max_ratio, window_size=self.window_size)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, n_kv, k_len, _ = keys.shape
        n_group = module.config.num_attention_heads // n_kv
        w = self.window_size
        if attentions is not None:
            attn = attentions[..., -w:, :]
        else:
            attn = _window_attention(module, hidden_states, keys, w, kwargs["position_embeddings"])

        head_attn = attn.mean(dim=-2)[0]
        p = head_attn / (head_attn.sum(dim=-1, keepdim=True) + 1e-9)
        logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
        entropy = -(p * logp).sum(dim=-1)
        keep = self._rule._layer_keep_ratio(entropy)
        self.compression_ratio = 1.0 - keep
        if hasattr(self, "_budget_log"):
            self._budget_log.append(keep)

        scores = attn.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        scores = scores.view(bsz, n_kv, n_group, k_len).mean(2)
        scores[..., -w:] = scores.max().item() + 1
        return scores
