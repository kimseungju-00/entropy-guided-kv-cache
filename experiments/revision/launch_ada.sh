#!/bin/bash
# Layer-wise budget + Ada-KV head allocation (M1Ada_e*) and Ada-SnapKV with per-sample
# scores (AdaSnap_*), on Mistral-7B and Llama-3.1-8B (Ada-SnapKV is not run on Qwen3-8B).
# Results: results/revision_phase1/<model>_ada.json
cd "$(dirname "$0")/../.."
export LONGBENCH_DATA=${LONGBENCH_DATA:-$PWD/data/longbench/data}
export HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/revision_phase1
CFG="M1Ada_e0.5 AdaSnap_0.2 M1Ada_e0.3 AdaSnap_0.1 M1Ada_e1.0 AdaSnap_0.3 M1Ada_e0.2 AdaSnap_0.15 M1Ada_e2.0 AdaSnap_0.4"
run() { CUDA_VISIBLE_DEVICES=$1 python experiments/revision/run_phase1.py --model $2 --configs $CFG --tag ada > $L/ada_$(basename $2).log 2>&1; }
( while [ ! -f $L/abs_Qwen3-8B.log ] || pgrep -f "Qwen/Qwen3-8B --configs Abs" >/dev/null; do sleep 120; done
  run 0 mistralai/Mistral-7B-Instruct-v0.3 ) &
( while [ ! -f $L/abs_Llama-3.1-8B-Instruct.log ] || pgrep -f "Llama-3.1-8B-Instruct --configs Abs" >/dev/null; do sleep 120; done
  run 1 meta-llama/Llama-3.1-8B-Instruct ) &
wait
