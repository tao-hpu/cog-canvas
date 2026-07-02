#!/usr/bin/env bash
# P1-b: run-to-run variance for the two headline LoCoMo cells (Cat 1-3, 699q).
# 3 fresh repeats each, headline protocol from .env (extractor=gpt-4o-mini,
# answerer=gpt-4o, judge=gpt-4o-mini). Chunks have no write-time LLM and no
# answer cache, so every repeat is live; artifacts get a fresh extraction-cache
# namespace per repeat via EXTRACTION_PROMPT_VARIANT (any value other than
# "clean"/"default" changes only the cache hash, not the prompt -- see
# get_extraction_config_hash in runner_locomo.py).
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
PY=/Users/TaoTao/opt/anaconda3/bin/python
OUT=experiments/results/variance_reruns; mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "=== P1-b variance runs, $(date) ===" > "$LOG"
for rep in 1 2 3; do
  echo ">>> [$(date +%H:%M:%S)] chunks rep$rep" | tee -a "$LOG"
  "$PY" -m experiments.runner_locomo -a cogcanvas-chunks-nograph --categories 1,2,3 \
      --llm-score -w 10 -o "$OUT/chunks_rep${rep}.json" >> "$LOG" 2>&1
  echo "<<< rc=$? chunks rep$rep $(date +%H:%M:%S)" | tee -a "$LOG"
  echo ">>> [$(date +%H:%M:%S)] artifacts rep$rep (fresh extraction ns vrep$rep)" | tee -a "$LOG"
  EXTRACTION_PROMPT_VARIANT="vrep${rep}" "$PY" -m experiments.runner_locomo -a cogcanvas --categories 1,2,3 \
      --llm-score -w 10 -o "$OUT/artifacts_rep${rep}.json" >> "$LOG" 2>&1
  echo "<<< rc=$? artifacts rep$rep $(date +%H:%M:%S)" | tee -a "$LOG"
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
touch "$OUT/done_ALL.flag"
