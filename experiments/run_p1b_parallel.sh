#!/usr/bin/env bash
# P1-b variance runs, parallel layout (4 lanes; embedding/reranker are a remote
# service so lanes only contend on API rate limits, retries handle 429s).
# chunks_rep1.json was already produced by the sequential attempt; artifacts
# vrep1 resumes its per-conversation extraction cache.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
PY=/Users/TaoTao/opt/anaconda3/bin/python
OUT=experiments/results/variance_reruns
LOG="$OUT/parallel.log"
echo "=== P1-b parallel lanes, $(date) ===" > "$LOG"

run_artifacts() {
  EXTRACTION_PROMPT_VARIANT="vrep$1" "$PY" -m experiments.runner_locomo -a cogcanvas \
    --categories 1,2,3 --llm-score -w 10 -o "$OUT/artifacts_rep$1.json" \
    > "$OUT/lane_artifacts_rep$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] artifacts rep$1 rc=$?" >> "$LOG"
}
run_chunks() {
  "$PY" -m experiments.runner_locomo -a cogcanvas-chunks-nograph \
    --categories 1,2,3 --llm-score -w 10 -o "$OUT/chunks_rep$1.json" \
    > "$OUT/lane_chunks_rep$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] chunks rep$1 rc=$?" >> "$LOG"
}

run_artifacts 1 &
run_artifacts 2 &
run_artifacts 3 &
( run_chunks 2; run_chunks 3 ) &
wait
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
touch "$OUT/done_ALL.flag"
