#!/bin/bash
# Quick ablation test to verify configurations

echo "========================================"
echo "Running Ablation Configuration Tests"
echo "========================================"

# Test 1: Full SOTA
echo ""
echo "Test 1: cogcanvas (Full SOTA - with all components)"
python -m experiments.runner_locomo \
  --agent cogcanvas \
  --llm-score \
  --samples 2 \
  --workers 1 \
  --verbose \
  --output experiments/results/test_cogcanvas_full.json

# Test 2: No Temporal
echo ""
echo "Test 2: cogcanvas-no-temporal (Remove Temporal)"
python -m experiments.runner_locomo \
  --agent cogcanvas-no-temporal \
  --llm-score \
  --samples 2 \
  --workers 1 \
  --verbose \
  --output experiments/results/test_cogcanvas_no_temporal.json

# Test 3: No Reranker
echo ""
echo "Test 3: cogcanvas-no-rerank (Remove Reranker)"
python -m experiments.runner_locomo \
  --agent cogcanvas-no-rerank \
  --llm-score \
  --samples 2 \
  --workers 1 \
  --verbose \
  --output experiments/results/test_cogcanvas_no_rerank.json

echo ""
echo "========================================"
echo "Summary of Results"
echo "========================================"

# Extract and display agent names from results
for f in experiments/results/test_cogcanvas_*.json; do
    if [ -f "$f" ]; then
        agent_name=$(python -c "import json; print(json.load(open('$f'))['agent_name'])" 2>/dev/null)
        accuracy=$(python -c "import json; print(json.load(open('$f'))['summary']['overall_accuracy'])" 2>/dev/null)
        echo "$(basename $f): $agent_name - Accuracy: $accuracy"
    fi
done
