#!/usr/bin/env bash
# M3 granularity-vs-fidelity control: 3 arms, full LoCoMo Cat 1-3, same batch.
# Answerer = gpt-4o-mini (ordering is answerer-invariant; ~10x cheaper).
set -u
cd "$(dirname "$0")/.." || exit 1
export ANSWER_MODEL=gpt-4o-mini
export BUILDER_MODEL=gpt-4o-mini
export LLMLINGUA_DEVICE=cpu
export PYTHONPATH=.
PY=/Users/TaoTao/opt/anaconda3/bin/python
OUT=experiments/results/m3_full; mkdir -p "$OUT"
LOG="$OUT/run.log"; echo "=== M3 full run @gpt-4o-mini, $(date) ===" > "$LOG"
for a in cogcanvas-chunks-nograph cogcanvas-no-graph cogcanvas-sent-nograph; do
  echo ">>> [$(date +%H:%M:%S)] $a" | tee -a "$LOG"
  t0=$(date +%s)
  "$PY" -m experiments.runner_locomo -a "$a" --categories 1,2,3 --llm-score -w 10 \
      -o "$OUT/${a}.json" >> "$LOG" 2>&1
  echo "<<< rc=$? $a in $(($(date +%s)-t0))s" | tee -a "$LOG"
  grep -E "overall_accuracy|single_hop_accuracy|temporal_accuracy|multi_hop_accuracy" "$OUT/${a}.json" 2>/dev/null | head -4 | tee -a "$LOG"
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
