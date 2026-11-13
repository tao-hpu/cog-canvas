#!/usr/bin/env bash
# EACL 2027 camera-ready (2026-11-13): same-code three-way Category-5 abstention table.
# Re-runs the chunks and artifacts arms on categories 1,2,3,5 under the current code so they are
# commensurate with the 2026-09-22 union run (locomo_union_10_cat1235.json, cache 7c39e0a4).
# Usage: run_cr_same_store.sh chunks | artifacts | artifacts_union_subset
#   artifacts_union_subset runs the artifact arm on the exact artifact subset of the union store
#   (built by experiments/build_union_artifact_subset.py), so union = chunks + artifacts object for object.
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
export EMBEDDING_API_BASE=http://127.0.0.1:9997/v1 RERANKER_API_BASE=http://127.0.0.1:9997/v1   # 4090 Xinference via SSH tunnel
export GRAPHRAG_EMBEDDING_API_BASE=$EMBEDDING_API_BASE
export LLMLINGUA_DEVICE=cpu PYTHONPATH=.
OUT=experiments/results/camera_ready; mkdir -p "$OUT"
case "${1:-}" in
  chunks)    name=chunks_nograph_cat1235;  agent=cogcanvas-chunks-nograph ;;
  artifacts) name=artifacts_graph_cat1235; agent=cogcanvas ;;
  artifacts_union_subset) name=artifacts_unionsubset_cat1235; agent=cogcanvas
     export EXTRACTION_CACHE_DIR_OVERRIDE=experiments/cache/extraction/union7c39e0a4_artifact_subset ;;
  union) name=union_cat1235; agent=cogcanvas-union ;;   # same-batch re-run of the 2026-09 union arm (cache 7c39e0a4)
  *) echo "usage: $0 chunks|artifacts|artifacts_union_subset|union"; exit 2 ;;
esac
echo ">>> $(date) $name agent=$agent ANSWER=$ANSWER_MODEL EXTRACTOR=$EXTRACTOR_MODEL" | tee -a "$OUT/run.log"
.venv/bin/python -m experiments.runner_locomo -a "$agent" --categories 1,2,3,5 --llm-score -w 10 -o "$OUT/$name.json" > "$OUT/$name.log" 2>&1
echo "<<< $(date) $name rc=$?" | tee -a "$OUT/run.log"
