# CogCanvas

> **Verbatim-Grounded Artifact Extraction for Long LLM Conversations**

[中文版](./README_CN.md) | English

[![arXiv](https://img.shields.io/badge/arXiv-2601.00821-b31b1b.svg)](https://arxiv.org/abs/2601.00821)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

CogCanvas is a **training-free framework** for maintaining long-term memory in LLM conversations. Inspired by how teams use whiteboards to anchor shared knowledge, CogCanvas extracts **verbatim-grounded cognitive artifacts** (Decisions, Facts, Todos) and organizes them into a queryable graph structure.

**The Problem**: Conversation summarization loses critical details. When asked "What coding style did we agree on?", summarization recalls "use type hints" but drops the constraint "**everywhere**" (19.0% exact match vs. 93.0% for CogCanvas).

**Our Solution**: Extract structured artifacts with exact source quotations, enabling traceable and hallucination-tolerant retrieval.

## Key Results

### LoCoMo Benchmark (Binary LLM-as-Judge)

| Method | Overall | Single-hop | Temporal | Multi-hop |
|--------|---------|------------|----------|-----------|
| **CogCanvas** | **32.4%** | **26.6%** | **32.7%** | **41.7%** |
| RAG | 24.6% | 24.6% | 12.1% | 40.6% |
| GraphRAG | 10.6% | 12.8% | 3.1% | 29.2% |
| Summarization | 5.6% | 5.3% | 0.6% | 22.9% |

**Key Finding**: CogCanvas achieves the highest overall accuracy among training-free methods:
- **Overall**: +7.8pp vs RAG (32.4% vs 24.6%)
- **Temporal reasoning**: +20.6pp vs RAG (32.7% vs 12.1%)
- **Multi-hop questions**: +1.1pp vs RAG (41.7% vs 40.6%)
- **Single-hop retrieval**: +2.0pp vs RAG (26.6% vs 24.6%)

### Controlled Benchmark

| Metric | Summarization | RAG | **CogCanvas** |
|--------|---------------|-----|---------------|
| Recall | 19.0% | 89.5% | **97.5%** |
| Exact Match | 19.0% | 89.5% | **93.0%** |
| Multi-hop Pass | 55.5% | 55.5% | **81.0%** |

## Quick Start

### Installation

```bash
git clone https://github.com/tao-hpu/cog-canvas.git
cd cog-canvas
pip install -e .
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1
EXTRACTOR_MODEL=gpt-4o-mini
ANSWER_MODEL=gpt-4o-mini
```

### Basic Usage

```python
from cogcanvas import Canvas

# Initialize
canvas = Canvas()

# Extract from dialogue
canvas.extract(
    user="Let's use PostgreSQL for the database. Budget is $500/month.",
    assistant="Good choice. I'll note the budget constraint."
)

# ... 50 turns later ...

# Retrieve with graph expansion
results = canvas.retrieve("Why PostgreSQL? What was our budget?")
# -> Returns linked artifacts: [DECISION: PostgreSQL] <-caused_by- [KEY_FACT: $500 budget]

# Inject into prompt
context = canvas.inject(results)
```

## Core Concepts

### Artifact Types

CogCanvas extracts 5 types of cognitive artifacts:

| Type | Description | Example |
|------|-------------|---------|
| `DECISION` | Choices made | "Use PostgreSQL for database" |
| `KEY_FACT` | Important facts | "Budget: $500/month" |
| `TODO` | Action items | "Set up database migrations" |
| `REMINDER` | Constraints | "Must support JSONB queries" |
| `INSIGHT` | Conclusions | "PostgreSQL fits our needs" |

### Verbatim Grounding

Each artifact includes a `quote` field with the exact source text:

```json
{
  "type": "decision",
  "content": "Use PostgreSQL as the primary database",
  "quote": "Let's use PostgreSQL for the database",
  "turn_id": 12
}
```

This enables:
- **Traceability**: Know exactly where information came from
- **Hallucination resistance**: Verify against original text
- **Compression tolerance**: Preserve details that summarization loses

### Graph-Enhanced Retrieval

Artifacts are connected with typed relationships:

```
[DECISION: Use PostgreSQL]
    ├── caused_by → [KEY_FACT: $500 budget]
    ├── caused_by → [REMINDER: Need JSONB support]
    └── leads_to → [TODO: Set up migrations]
```

Retrieval expands through the graph (default: 3 hops), surfacing related context that simple vector search misses.

## Project Structure

```
cog-canvas/
├── cogcanvas/              # Core library
│   ├── canvas.py           # Main Canvas class
│   ├── models.py           # Data models
│   ├── graph.py            # Graph operations
│   ├── temporal.py         # Temporal enhancement
│   └── llm/                # LLM backends
├── experiments/            # Evaluation code
│   ├── runner_locomo.py    # LoCoMo benchmark
│   └── agents/             # Agent implementations
├── web/                    # Web UI (FastAPI + Next.js)
├── EXPERIMENTS.md          # Reproduction guide
└── README.md
```

## Experiments

See [EXPERIMENTS.md](./EXPERIMENTS.md) for full reproduction instructions.

### Quick Test

```bash
# Run CogCanvas on 10 samples
python -m experiments.runner_locomo --agent cogcanvas --samples 10 --llm-score

# Compare with RAG baseline
python -m experiments.runner_locomo --agent rag --samples 10 --llm-score
```

### Ablation Studies

```bash
# Remove BGE reranking (largest impact: -11.5pp)
python -m experiments.runner_locomo --agent cogcanvas-no-rerank --llm-score

# Remove graph expansion (-6.6pp)
python -m experiments.runner_locomo --agent cogcanvas-no-graph --llm-score
```

## Web UI

### Start Backend (Port 3801)

```bash
cd web/backend
pip install -r requirements.txt
python main.py
```

### Start Frontend (Port 3800)

```bash
cd web/frontend
pnpm install && pnpm dev
```

Open [http://localhost:3800](http://localhost:3800)

## API Reference

### Canvas Class

```python
from cogcanvas import Canvas, ObjectType

# Initialize with custom models
canvas = Canvas(
    extractor_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small"
)

# Extract artifacts from dialogue
canvas.extract(user="...", assistant="...")

# Retrieve relevant artifacts
results = canvas.retrieve(
    query="Why did we choose X?",
    top_k=5,
    expand_hops=3  # Graph expansion depth
)

# Filter by type
decisions = canvas.retrieve(query, obj_type=ObjectType.DECISION)

# Format for prompt injection
context = canvas.inject(results, format="markdown")
```

## Citation

If you use CogCanvas in your research, please cite:

```bibtex
@article{an2026cogcanvas,
  title={CogCanvas: Verbatim-Grounded Artifact Extraction for Long LLM Conversations},
  author={An, Tao},
  journal={arXiv preprint arXiv:2601.00821},
  year={2026},
  url={https://arxiv.org/abs/2601.00821}
}
```

## License

MIT License - see [LICENSE](./LICENSE)

---

*CogCanvas: Because your AI shouldn't forget what you decided together.*
