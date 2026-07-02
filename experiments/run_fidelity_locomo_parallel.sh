#!/usr/bin/env bash
# Parallel tail of the LoCoMo fidelity run: the 4 remaining anchors at once.
# rag + secom already done sequentially; these 4 are independent jobs.
set -u
cd "$(dirname "$0")/.." || exit 1

export BUILDER_MODEL=gpt-4o
export LLMLINGUA_DEVICE=cpu
PY=/Users/TaoTao/opt/anaconda3/bin/python

OUT=experiments/results/fidelity_locomo
mkdir -p "$OUT"
LOG="$OUT/run_parallel.log"
echo "=== parallel tail started $(date) ===" >"$LOG"

# -w 6 each so total concurrency across 4 agents stays ~24 (rate-limit safe).
AGENTS=(summarization mem0 amem artifacts-flat)
pids=()
for a in "${AGENTS[@]}"; do
  echo ">>> launching $a @$(date +%H:%M:%S)" >>"$LOG"
  ( t0=$(date +%s)
    "$PY" -m experiments.runner_locomo -a "$a" --llm-score -w 6 \
        -o "$OUT/${a}_gpt4o.json" >"$OUT/${a}.log" 2>&1
    echo "<<< $a done rc=$? in $(( $(date +%s)-t0 ))s" >>"$LOG"
  ) &
  pids+=($!)
done

wait "${pids[@]}"
echo "=== ALL 4 DONE $(date) ===" >>"$LOG"
