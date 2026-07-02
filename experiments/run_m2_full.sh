#!/usr/bin/env bash
# M2 cross-lingual: chunks vs artifacts vs naive-RAG on PerLTQA (native Chinese).
set -u
cd "$(dirname "$0")/.." || exit 1
DS="experiments/data/perltqa/perltqa.json"
export ANSWER_MODEL=gpt-4o-mini BUILDER_MODEL=gpt-4o-mini LLMLINGUA_DEVICE=cpu PYTHONPATH=.
PY=/Users/TaoTao/opt/anaconda3/bin/python
OUT=experiments/results/m2_full; mkdir -p "$OUT"
LOG="$OUT/run.log"; echo "=== M2 PerLTQA full @mini, $(date) ===" > "$LOG"
for a in cogcanvas-chunks-nograph cogcanvas-no-graph rag; do
  echo ">>> [$(date +%H:%M:%S)] $a" | tee -a "$LOG"
  t0=$(date +%s)
  "$PY" -m experiments.runner_locomo -a "$a" --dataset "$DS" --max-questions 30 --categories 1,2,3 \
      --llm-score -w 10 -o "$OUT/${a}.json" >> "$LOG" 2>&1
  echo "<<< rc=$? $a in $(($(date +%s)-t0))s" | tee -a "$LOG"
  grep -E '"overall_accuracy"|"num_conversations"' "$OUT/${a}.json" 2>/dev/null | head -2 | tee -a "$LOG"
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
