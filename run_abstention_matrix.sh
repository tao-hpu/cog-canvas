#!/bin/zsh
# Tier-2 abstention run matrix. All gpt-4o-mini answerer, cat 1,2,3,5, 10 convs.
# Extraction cache (chunks-512, graph ON) is reused; only QA runs.
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas
PY=/Users/TaoTao/opt/anaconda3/bin/python3
export ANSWER_MODEL=gpt-4o-mini
export SCORE_MODEL=gpt-4o-mini
# zsh array so words split correctly when expanded
COMMON=(--samples 10 --categories 1,2,3,5 --llm-score --workers 10)
R=experiments/results

echo "===== [1/5] no-gate baseline (gpt-4o-mini) ====="
$PY -m experiments.runner_locomo --agent cogcanvas-chunks $COMMON \
    -o $R/locomo_chunks_mini_cat1235.json

echo "===== [2/5] M1 llm t=0.0 ====="
ABSTAIN_MODE=llm ABST_THRESHOLD=0.0 $PY -m experiments.runner_locomo \
    --agent cogcanvas-chunks-refuser $COMMON \
    -o $R/locomo_refuser_llm_t0.0_cat1235.json

echo "===== [3/5] M1 llm t=0.1 ====="
ABSTAIN_MODE=llm ABST_THRESHOLD=0.1 $PY -m experiments.runner_locomo \
    --agent cogcanvas-chunks-refuser $COMMON \
    -o $R/locomo_refuser_llm_t0.1_cat1235.json

echo "===== [4/5] M1 llm t=0.2 ====="
ABSTAIN_MODE=llm ABST_THRESHOLD=0.2 $PY -m experiments.runner_locomo \
    --agent cogcanvas-chunks-refuser $COMMON \
    -o $R/locomo_refuser_llm_t0.2_cat1235.json

echo "===== [5/5] M2 verify t=0.0 ====="
ABSTAIN_MODE=verify ABST_THRESHOLD=0.0 $PY -m experiments.runner_locomo \
    --agent cogcanvas-chunks-refuser $COMMON \
    -o $R/locomo_refuser_verify_cat1235.json

echo "===== MATRIX DONE ====="
