#!/usr/bin/env bash
# Fidelity curve re-run with a FIXED CONTEXT BUDGET (chars fed to answerer),
# not top_k -- removes the retrieval-budget confound. All 6 anchors, parallel.
set -u
cd "$(dirname "$0")/.." || exit 1

export BUILDER_MODEL=gpt-4o
export LLMLINGUA_DEVICE=cpu
export CONTEXT_BUDGET_CHARS=4000          # the controlled variable now held equal
PY=/Users/TaoTao/opt/anaconda3/bin/python

OUT=experiments/results/fidelity_locomo_budget
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== fidelity budget run (4000 chars) started $(date) ===" >"$LOG"

AGENTS=(rag secom summarization mem0 amem artifacts-flat)
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
echo "=== ALL 6 DONE $(date) ===" >>"$LOG"
