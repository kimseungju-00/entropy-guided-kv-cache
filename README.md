# Entropy-Guided Layer-Adaptive KV Cache Compression

Single-pass KV cache compression for long-context LLMs. Tokens are selected by
their window-averaged attention (SnapKV-style), and the retention ratio of each
layer is set from the entropy profile of the heads in that layer, with one
strength hyperparameter (epsilon). This repository holds the code and scripts
of the paper "Entropy-Guided Layer-Adaptive KV Cache Compression: A Controlled
Analysis of Token Selection and Per-Layer Allocation"; the result files are
archived on Zenodo (see Archive below).

The design rationale is an elementary identity: for a single attention head,
the total variation distance between the attention distributions before and
after eviction (with renormalization) equals the discarded attention mass
`delta`, and `delta` correlates with attention entropy.

## Key results

All comparisons use one matching rule (`src/eval/matching.py`: isotonic fit,
linear interpolation, all points, no extrapolation) with paired bootstrap 95%
confidence intervals over per-sample scores (10,000 resamples).

- **Allocation.** With token selection held fixed, the entropy-based per-layer
  allocation discards less attention mass than a uniform retention ratio of the
  same press, but its accuracy could not be distinguished from that ratio on any
  of the three models.
- **Token selection.** The accuracy advantage over SnapKV at equal budget appears
  to come mainly from the 32-token observation window of the press (SnapKV default:
  64). At equal windows the press could not be distinguished from SnapKV, although
  the method remained more accurate on Llama-3.1-8B. This does not generalize: a
  shorter window did not consistently improve PyramidKV or Ada-SnapKV.
- **Memory.** Savings relative to SnapKV have intervals that include zero on all
  three models, and the Qwen3-8B comparison is inconclusive.
- **CAKE.** CAKE is more accurate on Mistral-7B and Llama-3.1-8B through its
  allocation and token score combined rather than either alone.
- **Latency.** Single-pass; latency is within 0.1 s of SnapKV across context lengths.

## Method

`src/methods/proposed_kvpress.py` defines two presses on top of NVIDIA
[kvpress](https://github.com/NVIDIA/kvpress):

- `ProposedOnePassPress` — role-weighted budget (with head-role classification): the role-weighted variant of the component analysis.
- `EntropyBudgetPress` — **final method** (the "proposed method" of the paper,
  `Proposed_M1` in the result files): mean of the within-layer normalized head
  entropies (no head classification).

## Archive

The code together with all result files under `results/` (per-sample scores,
realized budgets, tables and figures) is archived on Zenodo:
https://doi.org/10.5281/zenodo.24133001. The `results/` directory is not
tracked in this repository; download the archive to reproduce the tables and
figures without rerunning the experiments.

## Setup

```bash
pip install -r requirements.txt
```

Download the [LongBench v1](https://github.com/THUDM/LongBench) `.jsonl` files
and either place them under `data/longbench/data/` or set:

```bash
export LONGBENCH_DATA=/path/to/longbench/data
```

Models used: `mistralai/Mistral-7B-Instruct-v0.3`,
`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`.

## Reproduce

Run from the project root. Results are written under `results/`.

```bash
# Theory: TV=delta and delta-entropy correlation, layer entropy heterogeneity
python experiments/theory/verify_delta.py --model <model> --n 15
python experiments/theory/measure_layer_entropy.py --model <model> --n 20

# Accuracy: baselines, then the final method (merged into the same JSON)
python experiments/performance/run_performance.py --model <model> --n 100
python experiments/performance/run_performance_m1.py --model <model> --n 100

# Memory at matched accuracy (no GPU; reads the accuracy JSON)
python experiments/memory/measure_memory_m1_final.py
python experiments/memory/compare_full_m1.py

# Latency / memory by context length
python experiments/efficiency/measure_efficiency_m1.py --model <model>

# Ablation and supplementary analysis
python experiments/ablation/run_ablation.py --model <model> --n 100
python experiments/analysis/diagnose_qwen.py --model <model> --n 15

# Controlled comparisons in experiments/revision (per-sample scores; one JSON per model and tag)
CUDA_VISIBLE_DEVICES=0 python experiments/revision/run_phase1.py --model <model>   # proposed vs uniform
#   other configurations (tags cake, ada, snap, pyr, abs, decomp, norm, press, w32) are listed in
#   experiments/revision/launch_queue.sh and run with --configs ... --tag <tag>
python experiments/revision/analyze_phase1.py --boot 10000                     # all paired comparisons
python experiments/revision/make_revision_outputs.py --boot 10000              # Figs. 3-4, Tables 6, A1-A3
python experiments/revision/plot_fig5.py                                        # Fig. 5
python experiments/revision/measure_layer_delta.py --model <model>              # per-layer delta (tag layerdelta)
python experiments/revision/analyze_layer_delta.py --boot 10000                 # phi-delta correlations, rule vs uniform delta
python experiments/revision/analyze_allocation_benefit.py --boot 10000         # worst-layer delta, budget at equal delta, normalization properties
python experiments/revision/analyze_sensitivity.py                              # mean saving under other matching rules

# Numbers quoted in the text of the paper, and its Tables 7-8
python figures_tables/compute_paper_numbers.py --boot 10000   # results/tables/paper_numbers.json
python figures_tables/make_comparison_tables.py                # results/tables/table7_comparisons.*, table8_comparisons.*

# Figs. 1-2 (results/figures) and the tables carried over from the original submission
python figures_tables/make_figures.py
python figures_tables/make_tables.py                            # tables/ (Tables 1-5, 9-11, A4-A6)
python figures_tables/make_budget_curve.py
```

### Where each item of the paper comes from

| Item | Script | Output |
|---|---|---|
| Table 1, Fig. 1 | `experiments/theory/verify_delta.py`, `figures_tables/make_figures.py` | `results/kvpress_theory/*_delta.json`, `results/figures/fig1_delta_entropy.png` |
| Table 2, Fig. 2 | `experiments/theory/measure_layer_entropy.py`, `figures_tables/make_figures.py` | `results/kvpress_theory/*_layerent.json`, `results/figures/fig2_layer_entropy.png` |
| Tables 3-5 | `experiments/performance/*`, `figures_tables/make_tables.py` | `results/kvpress_performance/*_perf.json` |
| Fig. 3, Fig. 4, Table 6, Tables A1-A3 | `experiments/revision/make_revision_outputs.py` | `results/figures/fig3_*`, `fig4_*`, `results/tables/table6_memory.*`, `tableA_*` |
| Fig. 5 | `experiments/revision/plot_fig5.py` | `results/figures/fig5_layer_allocation.png` |
| Tables 7-8 | `figures_tables/make_comparison_tables.py` | `results/tables/table7_comparisons.*`, `table8_comparisons.*` |
| Tables 9-11 | `experiments/ablation/run_ablation.py`, `figures_tables/make_tables.py` | `results/kvpress_ablation/*.json` |
| Tables A4-A6 | `experiments/efficiency/measure_efficiency_m1.py`, `figures_tables/make_tables.py` | `results/kvpress_efficiency_m1/*.json` |
| Numbers in the text | `figures_tables/compute_paper_numbers.py` | `results/tables/paper_numbers.json` |

## Layout

```
src/methods/proposed_kvpress.py   the two presses (final = EntropyBudgetPress)
src/eval/                         LongBench loader and official metrics
experiments/theory/               TV=delta, delta-entropy, layer entropy
experiments/performance/          accuracy (baselines + final method)
experiments/memory/               matched-accuracy KV memory
experiments/efficiency/           latency and memory by context length
experiments/ablation/             component analysis
experiments/analysis/             head-entropy analysis (Qwen3)
experiments/revision/             controlled comparisons, CAKE re-implementation, analysis, figures
src/eval/matching.py              matched-accuracy / matched-budget rule and bootstrap
figures_tables/                   figures, tables, and the numbers quoted in the paper
```

## Notes

- Qwen3-8B has a narrow accuracy range (about 0.16-0.18) and a non-monotonic
  accuracy-budget curve; all points are kept, and its memory comparison is
  inconclusive.
- `results/revision_phase1/*.json` hold the per-sample scores and realized
  budgets from which every table, figure and confidence interval of the paper
  is computed. The bootstrap seed is fixed, so `compute_paper_numbers.py --boot 10000`
  reproduces `results/tables/paper_numbers.json` exactly.
- Evaluation metrics and prompts are adapted from LongBench v1.

## License

Released under the Apache License 2.0; see `LICENSE`.
