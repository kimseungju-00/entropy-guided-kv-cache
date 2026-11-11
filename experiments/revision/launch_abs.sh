#!/bin/bash
# log-T normalization variant (Abs_e*) on the three models.
# Results: results/revision_phase1/<model>_abs.json
cd "$(dirname "$0")/../.."
export LONGBENCH_DATA=${LONGBENCH_DATA:-$PWD/data/longbench/data}
export HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/revision_phase1
CFG="Abs_e1.0 Abs_e0.5 Abs_e3.0 Abs_e0.7 Abs_e1.5"
run() { CUDA_VISIBLE_DEVICES=$1 python experiments/revision/run_phase1.py --model $2 --configs $CFG --tag abs > $L/abs_$(basename $2).log 2>&1; }
( while pgrep -f launch_cake.sh >/dev/null; do sleep 120; done
  run 1 mistralai/Mistral-7B-Instruct-v0.3; run 1 meta-llama/Llama-3.1-8B-Instruct ) &
( while pgrep -f launch_phase1.sh >/dev/null; do sleep 120; done
  run 0 Qwen/Qwen3-8B ) &
wait
