#!/usr/bin/env bash
# ARR Aug 2026 rebuttal add-ons for #69 (Fidelity). Usage:
#   run_rebuttal_arr_aug.sh smoke            # 5 questions, chunks arm, claude pair  (sanity: endpoint, keys, judge)
#   run_rebuttal_arr_aug.sh sent  [K]        # tEhS: sentence-verbatim at k=K (default 38) budget-matched to chunks' 2,127 tok/q; gpt-4o-mini like Table 10
#   run_rebuttal_arr_aug.sh claude [arm]     # jjMN: Table 1 pair under a non-OpenAI answerer/extractor; arm = chunks | artifacts | both (default both)
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a                                   # keys + uniapi bases from .env
export EMBEDDING_API_BASE=http://127.0.0.1:9997/v1 RERANKER_API_BASE=http://127.0.0.1:9997/v1   # 4090 Xinference via SSH tunnel
export GRAPHRAG_EMBEDDING_API_BASE=$EMBEDDING_API_BASE
export LLMLINGUA_DEVICE=cpu PYTHONPATH=.
PY=.venv/bin/python
OUT=experiments/results/rebuttal_arr_aug; mkdir -p "$OUT" experiments/cache   # cache dir pre-created: parallel workers race on os.mkdir
COMMON=(--categories 1,2,3 --llm-score -w 10)
mode=${1:-smoke}
run() { # name agent [extra...]
  local name=$1 agent=$2; shift 2
  echo ">>> [$(date +%H:%M:%S)] $name  (agent=$agent ANSWER=$ANSWER_MODEL EXTRACTOR=${EXTRACTOR_MODEL:-} BUILDER=${BUILDER_MODEL:-})" | tee -a "$OUT/run.log"
  local t0=$(date +%s)
  "$PY" -m experiments.runner_locomo -a "$agent" "${COMMON[@]}" "$@" -o "$OUT/$name.json" >> "$OUT/$name.log" 2>&1
  local rc=$?
  echo "<<< $name rc=$rc in $(( $(date +%s)-t0 ))s" | tee -a "$OUT/run.log"
  "$PY" - "$OUT/$name.json" <<'PYS' 2>/dev/null | tee -a "$OUT/run.log"
import json,sys; d=json.load(open(sys.argv[1])); s=d.get("summary",{})
print("   ", {k:s.get(k) for k in ("overall_accuracy","single_hop_accuracy","temporal_accuracy","multi_hop_accuracy","num_conversations")})
tu=d.get("token_usage") or {}; print("    token_usage:", {k:tu[k] for k in list(tu)[:6]} if isinstance(tu,dict) else tu)
PYS
}
case "$mode" in
  smoke)
    export ANSWER_MODEL=claude-sonnet-4-6 EXTRACTOR_MODEL=claude-haiku-4-5-20251001 BUILDER_MODEL=claude-haiku-4-5-20251001 MODEL_DEFAULT=claude-haiku-4-5-20251001
    run smoke_chunks_claude cogcanvas-chunks-nograph --max-questions 5 ;;
  sent)
    K=${2:-38}
    export ANSWER_MODEL=gpt-4o-mini BUILDER_MODEL=gpt-4o-mini EXTRACTOR_MODEL=gpt-4o-mini MODEL_DEFAULT=gpt-4o-mini
    export NOGRAPH_TOP_K=$K NOGRAPH_CAND_K=$((K*2))
    run "sent_k${K}_4omini" cogcanvas-sent-nograph ;;
  claude)
    arm=${2:-both}
    export ANSWER_MODEL=claude-sonnet-4-6 EXTRACTOR_MODEL=claude-haiku-4-5-20251001 BUILDER_MODEL=claude-haiku-4-5-20251001 MODEL_DEFAULT=claude-haiku-4-5-20251001
    [ "$arm" = chunks -o "$arm" = both ]    && run chunks_nograph_claude cogcanvas-chunks-nograph
    [ "$arm" = artifacts -o "$arm" = both ] && run artifacts_graph_claude cogcanvas ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo "=== $mode done $(date) ===" | tee -a "$OUT/run.log"
