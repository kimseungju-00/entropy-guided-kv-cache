#!/bin/bash
# Job queue for the remaining revision runs: each worker pops "<tag> <model>" lines from
# results/revision_phase1/queue.txt under flock.
# Usage: launch_queue.sh <gpu> [pid-to-wait-for]
cd "$(dirname "$0")/../.."
export LONGBENCH_DATA=${LONGBENCH_DATA:-$PWD/data/longbench/data}
export HF_HUB_DISABLE_PROGRESS_BARS=1
L=results/revision_phase1
Q=$L/queue.txt
GPU=$1
[ -n "$2" ] && while kill -0 "$2" 2>/dev/null; do sleep 60; done
cfgs() { # tag -> configs
  case $1 in
    cake) echo CAKE_0.2 CAKE_0.1 CAKE_0.3 CAKE_0.15 CAKE_0.4 ;;
    abs)  echo Abs_e1.0 Abs_e0.5 Abs_e3.0 Abs_e0.7 Abs_e1.5 ;;
    ada)  echo M1Ada_e0.5 AdaSnap_0.2 M1Ada_e0.3 AdaSnap_0.1 M1Ada_e1.0 AdaSnap_0.3 M1Ada_e0.2 AdaSnap_0.15 M1Ada_e2.0 AdaSnap_0.4 ;;
    snap) for k in 0.2 0.1 0.3 0.05 0.5 0.15; do echo -n "Snap_$k "; done ;;
    pyr)  for k in 0.2 0.1 0.3 0.05 0.5 0.15; do echo -n "Pyr_$k "; done ;;
    uniext) echo Uni_0.05 Uni_0.07 Uni_0.45 ;;  # extends the uniform curve; stored in the main file
    norm)   echo EntNorm_0.2 EntNorm_0.1 EntNorm_0.3 EntNorm_0.15 EntNorm_0.4 EntNormA4_0.2 EntNormA4_0.1 EntNormA4_0.3 ;;  # fixed-total variant
    press)  echo Snap32_0.2 UniW64_0.2 Snap32_0.1 UniW64_0.1 Snap32_0.3 UniW64_0.3 UniW16_0.2 UniW128_0.2 Snap32_0.15 UniW64_0.15 ;;  # press decomposition (window x implementation)
    pyr32)  for k in 0.2 0.1 0.3 0.05 0.5 0.15; do echo -n "Pyr32_$k "; done ;;  # PyramidKV, window 32
    ada32)  for k in 0.2 0.1 0.3 0.15 0.4; do echo -n "Ada32_$k "; done ;;  # Ada-SnapKV, window 32 (not Qwen3)
    decomp) echo M1Var_e0.5 CAKEMean_0.2 M1Var_e0.3 CAKEMean_0.1 M1Var_e1.0 CAKEMean_0.3 M1Var_e0.7 CAKEMean_0.15 M1Var_e2.0 CAKEMean_0.4 ;;
  esac
}
while true; do
  JOB=$(flock $Q.lock sh -c "head -n1 $Q; sed -i 1d $Q")
  [ -z "$JOB" ] && break
  set -- $JOB  # tag model
  echo "$(date '+%F %T') GPU$GPU start $1 $2" >> $L/queue.log
  TAG=$1; [ "$1" = uniext ] && TAG=""; case $1 in pyr32|ada32) TAG=w32 ;; esac
  if [ "$1" = layerdelta ]; then
    CUDA_VISIBLE_DEVICES=$GPU python experiments/revision/measure_layer_delta.py --model $2 > $L/$1_$(basename $2).log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$GPU python experiments/revision/run_phase1.py --model $2 --configs $(cfgs $1) --tag "$TAG" \
      > $L/$1_$(basename $2).log 2>&1
  fi
  echo "$(date '+%F %T') GPU$GPU done  $1 $2 (exit $?)" >> $L/queue.log
done
