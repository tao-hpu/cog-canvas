#!/bin/zsh
# LongMemEval-S recheck of the best LoCoMo mechanism (M1 llm t=0.0).
# Matched gpt-4o-mini answerer for BOTH no-gate baseline and refuser, so the
# refuser-vs-baseline delta is a clean same-answerer comparison. Extraction
# cache (chunks-nograph) reused; only QA runs.
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas
PY=/Users/TaoTao/opt/anaconda3/bin/python3
export ANSWER_MODEL=gpt-4o-mini
export SCORE_MODEL=gpt-4o-mini
R=experiments/results

echo "===== [1/2] LME-S no-gate chunks-nograph baseline (gpt-4o-mini) ====="
$PY -m experiments.runner_longmemeval --agent cogcanvas-chunks-nograph \
    --samples 500 --workers 10 -o $R/lme_s_chunks_nograph_mini_500.json

echo "===== [2/2] LME-S M1 refuser (llm t=0.0, gpt-4o-mini) ====="
ABSTAIN_MODE=llm ABST_THRESHOLD=0.0 $PY -m experiments.runner_longmemeval \
    --agent cogcanvas-chunks-refuser \
    --samples 500 --workers 10 -o $R/lme_s_refuser_llm_t0.0_500.json

echo "===== LME RECHECK DONE ====="
