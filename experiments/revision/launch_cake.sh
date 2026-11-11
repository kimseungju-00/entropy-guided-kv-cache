#!/bin/bash
# CAKE comparison on the three models.
# Results: results/revision_phase1/<model>_cake.json
cd "$(dirname "$0")/../.."
export LONGBENCH_DATA=${LONGBENCH_DATA:-$PWD/data/longbench/data}
export HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/revision_phase1
while pgrep -f "run_phase1.py --model meta-llama/Llama-3.1-8B-Instruct" >/dev/null; do sleep 120; done
CFG="CAKE_0.2 CAKE_0.1 CAKE_0.3 CAKE_0.15 CAKE_0.4"
for M in meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen3-8B; do
  CUDA_VISIBLE_DEVICES=1 python experiments/revision/run_phase1.py --model $M --configs $CFG --tag cake > $L/cake_$(basename $M).log 2>&1
done
