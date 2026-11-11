#!/bin/bash
# Proposed method and uniform ratio of the same press on the three models (two GPUs).
# Resumable: finished config/task pairs are skipped.
cd "$(dirname "$0")/../.."
export LONGBENCH_DATA=${LONGBENCH_DATA:-$PWD/data/longbench/data}
export HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/revision_phase1; mkdir -p $L
( CUDA_VISIBLE_DEVICES=0 python experiments/revision/run_phase1.py --model mistralai/Mistral-7B-Instruct-v0.3 > $L/mistral.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python experiments/revision/run_phase1.py --model Qwen/Qwen3-8B > $L/qwen.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=1 python experiments/revision/run_phase1.py --model meta-llama/Llama-3.1-8B-Instruct > $L/llama.log 2>&1 &
wait
