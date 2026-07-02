#!/usr/bin/env bash
# M3 budget-matched sentence arm: raise k so small sentence items fill >= the
# chunks@k15 answerer-token budget. If it still trails chunks, the chunks>sentence
# gap is genuine granularity, not retrieval budget.
set -u
cd "$(dirname "$0")/.." || exit 1
export ANSWER_MODEL=gpt-4o-mini BUILDER_MODEL=gpt-4o-mini LLMLINGUA_DEVICE=cpu PYTHONPATH=.
export NOGRAPH_TOP_K=90 NOGRAPH_CAND_K=180
PY=/Users/TaoTao/opt/anaconda3/bin/python
OUT=experiments/results/m3_full; mkdir -p "$OUT"
LOG="$OUT/budgetmatch.log"; echo "=== sentence budget-match k=90, $(date) ===" > "$LOG"
echo ">>> [$(date +%H:%M:%S)] sent@k90" | tee -a "$LOG"
t0=$(date +%s)
"$PY" -m experiments.runner_locomo -a cogcanvas-sent-nograph --categories 1,2,3 --llm-score -w 10 \
    -o "$OUT/sent_budgetmatch_k90.json" >> "$LOG" 2>&1
echo "<<< rc=$? in $(($(date +%s)-t0))s" | tee -a "$LOG"
grep -E "overall_accuracy|single_hop_accuracy|temporal_accuracy|multi_hop_accuracy" "$OUT/sent_budgetmatch_k90.json" 2>/dev/null | head -4 | tee -a "$LOG"
echo "=== DONE $(date) ===" | tee -a "$LOG"
