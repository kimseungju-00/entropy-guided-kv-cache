#!/bin/bash
# SnapKV and PyramidKV with per-sample scores at the budgets of the paper.
# Results: results/revision_phase1/<model>_snap.json, <model>_pyr.json
cd "$(dirname "$0")/../.."
export LONGBENCH_DATA=${LONGBENCH_DATA:-$PWD/data/longbench/data}
export HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/revision_phase1
K="0.2 0.1 0.3 0.05 0.5 0.15"
run() { # gpu model method
  CUDA_VISIBLE_DEVICES=$1 python experiments/revision/run_phase1.py --model $2 \
    --configs $(for k in $K; do echo -n "$3_$k "; done) --tag $(echo $3 | tr A-Z a-z) \
    > $L/$(echo $3 | tr A-Z a-z)_$(basename $2).log 2>&1; }
( while [ ! -f $L/ada_Mistral-7B-Instruct-v0.3.log ] || pgrep -f "Mistral-7B-Instruct-v0.3 --configs M1Ada" >/dev/null; do sleep 120; done
  run 0 Qwen/Qwen3-8B Snap; run 0 Qwen/Qwen3-8B Pyr; run 0 mistralai/Mistral-7B-Instruct-v0.3 Pyr ) &
( while [ ! -f $L/ada_Llama-3.1-8B-Instruct.log ] || pgrep -f "Llama-3.1-8B-Instruct --configs M1Ada" >/dev/null; do sleep 120; done
  run 1 meta-llama/Llama-3.1-8B-Instruct Snap; run 1 meta-llama/Llama-3.1-8B-Instruct Pyr; run 1 mistralai/Mistral-7B-Instruct-v0.3 Snap ) &
wait
