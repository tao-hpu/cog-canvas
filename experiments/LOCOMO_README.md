# LoCoMo Benchmark Integration

This directory contains the adapter and runner for evaluating CogCanvas on the LoCoMo (Long Context Multi-hop) benchmark.

## Overview

**LoCoMo** is a benchmark for evaluating long-context multi-hop question answering in conversational settings. It features:

- Multi-session conversations between two speakers
- Real-world dialogue with temporal and contextual complexity
- Question categories:
  - **Single-hop (Category 1)**: Direct fact retrieval
  - **Temporal (Category 2)**: Time-based reasoning
  - **Multi-hop (Category 3)**: Connecting multiple facts
  - **Category 4 & 5**: Additional question types

## Dataset Structure

### Raw LoCoMo Format

```json
{
  "qa": [
    {
      "question": "When did Caroline go to the LGBTQ support group?",
      "answer": "7 May 2023",
      "evidence": ["D1:3"],
      "category": 2
    }
  ],
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {
        "speaker": "Caroline",
        "dia_id": "D1:1",
        "text": "Hey Mel! Good to see you! How have you been?"
      }
    ]
  }
}
```

### Converted Format

The adapter converts LoCoMo to CogCanvas evaluation format:

- **Flattened turns**: Multi-session dialogues → sequential conversation turns
- **Speaker mapping**: Speaker A → user, Speaker B → assistant
- **Evidence tracking**: Dialogue IDs mapped to turn numbers
- **Compression point**: Set at conversation midpoint by default

## Files

### Core Components

- **`locomo_adapter.py`**: Converts LoCoMo data to CogCanvas format
- **`runner_locomo.py`**: Experiment runner for LoCoMo evaluation
- **`test_locomo.py`**: Test suite for adapter and runner

### Data Files

- **`data/locomo10.json`**: Raw LoCoMo dataset (10 conversations)
- **`data/locomo_converted.json`**: Converted format (generated)

## Usage

### 1. Convert Dataset

```bash
# Convert raw LoCoMo to CogCanvas format
python -m experiments.locomo_adapter \
  --input experiments/data/locomo10.json \
  --output experiments/data/locomo_converted.json \
  --verify

# Output:
# - Converted JSON file
# - Conversion statistics
# - Evidence mapping verification
```

### 2. Run Quick Test

```bash
# Test adapter and runner
python -m experiments.test_locomo

# Tests:
# - Data loading and conversion
# - Evidence mapping accuracy
# - Runner with sample agent
```

### 3. Evaluate Single Agent

```bash
# Evaluate CogCanvas (full config)
python -m experiments.runner_locomo \
  --agent cogcanvas \
  --samples 10 \
  --workers 4 \
  --output results/locomo_cogcanvas.json

# Evaluate baseline (native context window)
python -m experiments.runner_locomo \
  --agent native \
  --retain-recent 10 \
  --samples 10

# Evaluate RAG baseline
python -m experiments.runner_locomo \
  --agent rag \
  --retain-recent 5 \
  --samples 10
```

### 4. Evaluate All Agents (Ablation Study)

```bash
# Run all CogCanvas variants
for agent in cogcanvas cogcanvas-nograph cogcanvas-baseline \
             cogcanvas-temporal cogcanvas-hybrid cogcanvas-cot; do
  python -m experiments.runner_locomo \
    --agent $agent \
    --samples 10 \
    --workers 4 \
    --output "results/locomo_${agent}.json"
done

# Run baseline agents
for agent in native rag summarization memgpt-lite graphrag-lite; do
  python -m experiments.runner_locomo \
    --agent $agent \
    --samples 10 \
    --workers 4 \
    --output "results/locomo_${agent}.json"
done
```

## Command-Line Options

### `locomo_adapter.py`

```
--input, -i       Path to LoCoMo JSON file (default: experiments/data/locomo10.json)
--output, -o      Path to output file (default: experiments/data/locomo_converted.json)
--verify          Verify evidence ID mappings
```

### `runner_locomo.py`

```
--dataset, -d         Path to LoCoMo dataset (default: experiments/data/locomo10.json)
--agent, -a           Agent to evaluate (cogcanvas, native, rag, etc.)
--samples, -n         Number of conversations to evaluate (default: all)
--output, -o          Output JSON file for results
--compression-turn    Fixed compression turn (default: middle of conversation)
--retain-recent       Number of recent turns to retain (default: 5)
--workers, -w         Number of parallel workers (default: 1)
--max-questions       Max questions per conversation (for testing)
```

## Evaluation Metrics

### Primary Metrics

- **Accuracy**: Fraction of questions correctly answered (>60% keyword overlap or exact match)
- **Exact Match Rate**: Fraction with exact answer string match
- **Keyword Overlap**: Average keyword overlap between answer and ground truth

### Category Breakdown

- **Single-hop Accuracy**: Direct fact retrieval
- **Temporal Accuracy**: Time-based reasoning questions
- **Multi-hop Accuracy**: Questions requiring multiple facts

## Scoring Methodology

### Keyword Overlap

1. Extract keywords from ground truth answer
   - Tokenize and lowercase
   - Remove stop words
   - Filter short tokens (<2 chars)

2. Extract keywords from model answer
3. Compute overlap: `found_keywords / total_keywords`

### Pass Criteria

A question is considered **passed** if:
- Exact match: Ground truth appears in answer (case-insensitive), OR
- Keyword overlap ≥ 60%

## Dataset Statistics

From `locomo10.json`:

```
Total conversations: 10
Total turns: 2,938
Total QA pairs: 1,542
Average turns/conversation: 293.8
Average QA pairs/conversation: 154.2

Category Distribution:
  Single-hop (1): 282 questions (18.3%)
  Temporal (2): 321 questions (20.8%)
  Multi-hop (3): 96 questions (6.2%)
  Category 4: 841 questions (54.5%)
  Category 5: 2 questions (0.1%)
```

## Example Output

```
============================================================
LoCoMo Experiment: CogCanvas(graph=True, temporal=True)
Conversations: 10
Compression: middle
Retain recent: 5 turns
Max workers: 4
============================================================

[1/10] Conversation locomo_000
    Compression at turn 104/209
    => Accuracy: 65% | Exact: 42% | Overlap: 58%

[2/10] Conversation locomo_001
    ...

============================================================
LOCOMO RESULTS SUMMARY
============================================================
  agent: CogCanvas(graph=True, temporal=True)
  num_conversations: 10
  overall_accuracy: 68.5%
  exact_match_rate: 45.2%
  keyword_overlap: 62.3%
  single_hop_accuracy: 72.1%
  temporal_accuracy: 65.8%
  multi_hop_accuracy: 58.3%
```

## Implementation Details

### Conversation Flattening

Multi-session LoCoMo conversations are flattened into sequential turns:

```python
# Session 1
D1:1 (Caroline) → Turn 1 user
D1:2 (Melanie) → Turn 1 assistant
D1:3 (Caroline) → Turn 2 user
...

# Session 2
D2:1 (Melanie) → Turn N assistant
D2:2 (Caroline) → Turn N+1 user
...
```

### Evidence Mapping

Dialogue IDs (e.g., "D1:3") are mapped to turn numbers for:
- Analyzing retrieval accuracy
- Identifying which turns contain evidence
- Validating model reasoning

### Compression Strategy

- **Default**: Compress at conversation midpoint
- **Custom**: Specify fixed `--compression-turn`
- **Retained turns**: Keep last N turns before compression point

## Integration with CogCanvas Agents

All CogCanvas agents inherit from `Agent` base class:

```python
class Agent(ABC):
    def process_turn(self, turn: ConversationTurn) -> None:
        """Process conversation turn"""

    def answer_question(self, question: str) -> AgentResponse:
        """Answer recall question"""

    def on_compression(self, retained_turns: List[ConversationTurn]) -> None:
        """Handle compression event"""

    def reset(self) -> None:
        """Reset state between conversations"""
```

### Supported Agents

- **CogCanvas variants**: cogcanvas, cogcanvas-nograph, cogcanvas-baseline, etc.
- **Baseline agents**: native, rag, summarization, memgpt-lite, graphrag-lite

## Troubleshooting

### Missing Answers

Some QA pairs in LoCoMo don't have answers. The adapter automatically filters these out:

```python
qa_pairs = [
    LoCoMoQAPair(...)
    for qa in qa_data
    if 'answer' in qa and qa['answer']  # Skip missing answers
]
```

### Evidence Mapping Errors

A small number of dialogue IDs may not map correctly (typically <1%):
- Malformed IDs (e.g., "D:11:26" instead of "D11:26")
- References to non-existent sessions
- These are logged during `--verify` mode

### Memory Issues

For large evaluations:
- Use `--workers` for parallelization
- Limit with `--samples` during testing
- Use `--max-questions` to cap questions per conversation

## Citation

If you use LoCoMo in your research:

```bibtex
@inproceedings{locomo2024,
  title={LoCoMo: Long Context Multi-hop Question Answering},
  author={[Authors]},
  booktitle={[Conference]},
  year={2024}
}
```

## Future Enhancements

- [ ] Add evidence-based retrieval metrics
- [ ] Support category-specific evaluation
- [ ] Add temporal reasoning analysis
- [ ] Implement cross-session fact tracking
- [ ] Add visualization of question difficulty vs accuracy
