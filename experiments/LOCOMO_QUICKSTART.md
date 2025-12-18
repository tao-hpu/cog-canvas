# LoCoMo Quick Start Guide

## What is LoCoMo?

LoCoMo (Long Context Multi-hop) is a benchmark for evaluating long-context question answering with:
- Real-world multi-session conversations (avg 294 turns)
- 1,542 questions across multiple reasoning types
- Evidence-based evaluation

## Installation Verification

```bash
# Check that files exist
ls -l experiments/locomo_adapter.py
ls -l experiments/runner_locomo.py
ls -l experiments/data/locomo10.json

# Run quick test
python -m experiments.test_locomo
```

## Basic Usage

### 1. Run Test (30 seconds)

```bash
python -m experiments.test_locomo
```

Expected output:
```
============================================================
Testing LoCoMo Adapter
============================================================
Loaded 10 raw conversations
Converted 10 conversations
First conversation (locomo_000):
  Speakers: Caroline and Melanie
  Turns: 209
  QA pairs: 154
  Compression point: 104
```

### 2. Convert Dataset (Optional)

```bash
python -m experiments.locomo_adapter \
  --input experiments/data/locomo10.json \
  --output experiments/data/locomo_converted.json \
  --verify
```

### 3. Run Single Evaluation

```bash
# Quick test (1 conversation, 3 questions)
python -m experiments.runner_locomo \
  --agent native \
  --samples 1 \
  --max-questions 3

# Full evaluation on CogCanvas
python -m experiments.runner_locomo \
  --agent cogcanvas \
  --output results/locomo_cogcanvas.json
```

### 4. Parallel Evaluation

```bash
# Use 10 workers for speed
python -m experiments.runner_locomo \
  --agent cogcanvas \
  --workers 10 \
  --output results/locomo_cogcanvas_parallel.json
```

## Command Cheat Sheet

### Essential Commands

```bash
# Test implementation
python -m experiments.test_locomo

# Evaluate single agent
python -m experiments.runner_locomo --agent cogcanvas

# Fast test (2 conversations, 10 questions each)
python -m experiments.runner_locomo --agent native --samples 2 --max-questions 10

# Full parallel run
python -m experiments.runner_locomo --agent cogcanvas --workers 10 --output results/locomo.json

# Compare all agents
for agent in cogcanvas native rag summarization; do
  python -m experiments.runner_locomo --agent $agent --output results/locomo_$agent.json
done
```

### Available Agents

- `cogcanvas` - Full CogCanvas system (SOTA)
- `cogcanvas-nograph` - Without graph expansion
- `cogcanvas-baseline` - Baseline configuration
- `cogcanvas-temporal` - With temporal heuristic
- `cogcanvas-hybrid` - Hybrid retrieval
- `cogcanvas-cot` - With chain-of-thought
- `native` - Basic context window
- `rag` - Retrieval-augmented generation
- `summarization` - Summarization-based
- `memgpt-lite` - MemGPT-style memory
- `graphrag-lite` - Graph-based RAG

### Common Options

```bash
--agent, -a        Agent to evaluate (default: cogcanvas)
--samples, -n      Number of conversations (default: all 10)
--max-questions    Max questions per conversation
--workers, -w      Parallel workers (default: 1, recommend: 10)
--output, -o       Output JSON file path
--retain-recent    Recent turns to keep (default: 5)
--compression-turn Fixed compression point (default: middle)
```

## Understanding Results

### Sample Output

```bash
$ python -m experiments.runner_locomo --agent cogcanvas --samples 1 --max-questions 5

============================================================
LoCoMo Experiment: CogCanvas
Conversations: 1
============================================================

[1/1] Conversation locomo_000
    Compression at turn 104/209
      ✓ [temporal] When did Caroline go to LGBTQ... -> 100%
      ✓ [temporal] When did Melanie paint sunrise... -> 100%
      ✗ [multi-hop] What fields would Caroline... -> 50%
      ✓ [single-hop] What did Caroline research... -> 100%
      ✓ [single-hop] What is Caroline's identity... -> 100%
    => Accuracy: 80% | Exact: 60% | Overlap: 90%

============================================================
LOCOMO RESULTS SUMMARY
============================================================
  agent: CogCanvas
  num_conversations: 1
  overall_accuracy: 80.0%
  exact_match_rate: 60.0%
  keyword_overlap: 90.0%
  single_hop_accuracy: 100.0%
  temporal_accuracy: 100.0%
  multi_hop_accuracy: 50.0%
```

### Metrics Explained

- **Accuracy**: % questions passed (exact match OR keyword overlap >= 60%)
- **Exact Match Rate**: % with exact answer string in response
- **Keyword Overlap**: Average % of answer keywords found
- **Category Accuracy**: Accuracy by question type (single-hop, temporal, multi-hop)

### Question Categories

1. **Single-hop (18%)**: Direct fact retrieval
   - Example: "What is Caroline's identity?" → "Transgender woman"

2. **Temporal (21%)**: Time-based reasoning
   - Example: "When did Caroline go to the support group?" → "7 May 2023"

3. **Multi-hop (6%)**: Connect multiple facts
   - Example: "What fields would Caroline pursue?" → "Psychology, counseling"

4. **Category 4 (55%)**: Extended reasoning (LoCoMo-specific)

## Typical Workflows

### Quick Test Before Development

```bash
# Verify everything works (30 seconds)
python -m experiments.test_locomo

# Quick evaluation (2 minutes)
python -m experiments.runner_locomo --agent native --samples 1 --max-questions 10
```

### Development Testing

```bash
# Test your changes on small sample
python -m experiments.runner_locomo \
  --agent cogcanvas \
  --samples 2 \
  --max-questions 20

# Compare before/after
python -m experiments.runner_locomo --agent cogcanvas-baseline --samples 3 --output before.json
# ... make changes ...
python -m experiments.runner_locomo --agent cogcanvas --samples 3 --output after.json
```

### Full Evaluation

```bash
# Single agent, full dataset (10-20 minutes)
python -m experiments.runner_locomo \
  --agent cogcanvas \
  --workers 10 \
  --output results/locomo_cogcanvas_$(date +%Y%m%d).json

# All agents comparison (1-2 hours)
./experiments/scripts/run_locomo_all.sh
```

## Troubleshooting

### Problem: Command not found

```bash
# Make sure you're in the cog-canvas directory
cd /path/to/cog-canvas
python -m experiments.test_locomo
```

### Problem: Import errors

```bash
# Install dependencies
pip install -r requirements.txt

# Check Python path
export PYTHONPATH=/path/to/cog-canvas:$PYTHONPATH
```

### Problem: Slow evaluation

```bash
# Solution 1: Use parallel workers
python -m experiments.runner_locomo --agent cogcanvas --workers 10

# Solution 2: Limit scope
python -m experiments.runner_locomo --agent cogcanvas --samples 3 --max-questions 20

# Solution 3: Use faster agent for testing
python -m experiments.runner_locomo --agent native --samples 1
```

### Problem: Low accuracy

**Check compression point:**
```bash
# Default is middle - may be too early
python -m experiments.runner_locomo --agent cogcanvas --compression-turn 150

# Keep more recent context
python -m experiments.runner_locomo --agent cogcanvas --retain-recent 10
```

**Check individual results:**
```bash
# Save detailed output
python -m experiments.runner_locomo --agent cogcanvas --output results/debug.json

# Examine specific questions
cat results/debug.json | jq '.conversations[0].questions[] | {question, answer, passed}'
```

## File Locations

### Input Data

```
experiments/data/locomo10.json          # Original LoCoMo dataset (10 conversations)
experiments/data/locomo_converted.json  # Converted format (optional, auto-generated)
```

### Code Files

```
experiments/locomo_adapter.py    # Data loading and conversion
experiments/runner_locomo.py     # Evaluation runner
experiments/test_locomo.py       # Test suite
```

### Results

```
results/locomo_*.json            # Evaluation results
experiments/results/locomo_*/    # Batch evaluation results
```

## Next Steps

1. **Verify Setup**: Run `python -m experiments.test_locomo`
2. **Quick Test**: Test on 1 conversation
3. **Full Evaluation**: Run on all 10 conversations
4. **Compare Agents**: Evaluate multiple agents
5. **Analyze Results**: Generate comparison tables

## Getting Help

- **Full Documentation**: See `experiments/LOCOMO_GUIDE.md`
- **Code Reference**: See inline documentation in `locomo_adapter.py` and `runner_locomo.py`
- **Implementation Details**: Check `experiments/LOCOMO_README.md` and `LOCOMO_SUMMARY.md`

## Summary

LoCoMo integration is **complete and ready to use**:

- ✓ Data adapter implemented
- ✓ Runner implemented
- ✓ All agents supported
- ✓ Parallel execution ready
- ✓ Evidence tracking enabled
- ✓ Category-based analysis
- ✓ Full test suite

**Start evaluating now:**
```bash
python -m experiments.runner_locomo --agent cogcanvas
```
