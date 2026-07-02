#!/usr/bin/env bash
# Fidelity-curve full run on LoCoMo, builder = gpt-4o (steelman headline).
# 6 anchors, one flat retrieve->rerank->reason pipeline, only representation varies.
set -u
cd "$(dirname "$0")/.." || exit 1

export BUILDER_MODEL=gpt-4o          # steelman: strongest builder for every lossy anchor
export LLMLINGUA_DEVICE=cpu

# Pin the interpreter: a bare `python` resolves to MAMP python2.7 in the
# background shell, which cannot parse the (py3) codebase. Use anaconda3.
PY=/Users/TaoTao/opt/anaconda3/bin/python

OUT=experiments/results/fidelity_locomo
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== fidelity LoCoMo run @gpt-4o builder, started $(date) ===" >"$LOG"

# fidelity-curve order: verbatim -> SeCom -> summary -> Mem0 -> A-Mem -> artifacts
AGENTS=(rag secom summarization mem0 amem artifacts-flat)

for a in "${AGENTS[@]}"; do
  echo "" | tee -a "$LOG"
  echo ">>> [$(date +%H:%M:%S)] running $a ..." | tee -a "$LOG"
  t0=$(date +%s)
  "$PY" -m experiments.runner_locomo -a "$a" --llm-score -w 10 \
      -o "$OUT/${a}_gpt4o.json" >>"$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "<<< $a done rc=$rc in $((t1-t0))s" | tee -a "$LOG"
  grep -E "overall_accuracy|single_hop|temporal|multi_hop" "$OUT/${a}_gpt4o.json" 2>/dev/null | head -1 | tee -a "$LOG" || true
done

echo "" | tee -a "$LOG"
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
