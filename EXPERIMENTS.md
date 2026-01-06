# CogCanvas Experiment Reproduction Guide

This document provides commands to reproduce all experiments in the paper.

## Prerequisites

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Configuration (.env)
```bash
cp .env.example .env
# Edit .env with your API keys:
EXTRACTOR_MODEL=gpt-4o-mini   # Artifact extraction
ANSWER_MODEL=gpt-4o-mini      # Answer generation
SCORE_MODEL=gpt-4o-mini       # LLM Judge scoring
OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # or your proxy
```

---

## Main Experiments

**IMPORTANT**: Always use `--categories "1,2,3"` to filter to official LoCoMo categories (single-hop, temporal, multi-hop). Category 4/5 are not part of the official benchmark.

### CogCanvas (Full System)
```bash
python -m experiments.runner_locomo --agent cogcanvas --llm-score --categories "1,2,3"
```

### Baselines
```bash
# Native (no memory)
python -m experiments.runner_locomo --agent native --llm-score --categories "1,2,3"

# Summarization
python -m experiments.runner_locomo --agent summarization --llm-score --categories "1,2,3"

# RAG (chunk-based retrieval)
python -m experiments.runner_locomo --agent rag --llm-score --categories "1,2,3"

# GraphRAG (requires longer time for indexing)
python -m experiments.runner_locomo --agent graphrag --llm-score --workers 1 --categories "1,2,3"
```

### Run All Baselines
```bash
python -m experiments.runner_locomo --agent cogcanvas --llm-score --categories "1,2,3" && \
python -m experiments.runner_locomo --agent native --llm-score --categories "1,2,3" && \
python -m experiments.runner_locomo --agent summarization --llm-score --categories "1,2,3" && \
python -m experiments.runner_locomo --agent rag --llm-score --categories "1,2,3"
```

---

## Ablation Studies

Remove single component from full system to measure contribution.

**Paper Results (Cat 1/2/3, 10 conversations, LLM scoring)**:
- Full System: 28.6%
- w/o Gleaning: 30.7% (+2.1pp)
- w/o Graph: 25.8% (-2.8pp)
- w/o Reranker: 20.9% (-7.7pp)
- Minimal: 14.3% (-14.3pp)

```bash
# Full System
python -m experiments.runner_locomo --agent cogcanvas --llm-score --categories "1,2,3"

# Remove CoT prompting
python -m experiments.runner_locomo --agent cogcanvas-no-cot --llm-score --categories "1,2,3"

# Remove Temporal heuristic edges
python -m experiments.runner_locomo --agent cogcanvas-no-temporal --llm-score --categories "1,2,3"

# Remove Hybrid retrieval (semantic only)
python -m experiments.runner_locomo --agent cogcanvas-no-hybrid --llm-score --categories "1,2,3"

# Remove BGE Reranker (major contributor: -7.7pp)
python -m experiments.runner_locomo --agent cogcanvas-no-rerank --llm-score --categories "1,2,3"

# Remove Gleaning (second-pass extraction)
python -m experiments.runner_locomo --agent cogcanvas-no-gleaning --llm-score --categories "1,2,3"

# Remove Graph expansion
python -m experiments.runner_locomo --agent cogcanvas-no-graph --llm-score --categories "1,2,3"

# Minimal baseline (graph only, no enhancements)
python -m experiments.runner_locomo --agent cogcanvas-minimal --llm-score --categories "1,2,3"
```

### Ablation Configuration Reference

| Agent | Graph | Temporal | Retrieval | Prompt | Reranker | Gleaning |
|-------|-------|----------|-----------|--------|----------|----------|
| **cogcanvas** | ✅ | ✅ | hybrid | cot | ✅ | ✅ |
| cogcanvas-no-cot | ✅ | ✅ | hybrid | direct | ✅ | ✅ |
| cogcanvas-no-temporal | ✅ | ❌ | hybrid | cot | ✅ | ✅ |
| cogcanvas-no-hybrid | ✅ | ✅ | semantic | cot | ✅ | ✅ |
| cogcanvas-no-rerank | ✅ | ✅ | hybrid | cot | ❌ | ✅ |
| cogcanvas-no-gleaning | ✅ | ✅ | hybrid | cot | ✅ | ❌ |
| cogcanvas-no-graph | ❌ | ✅ | hybrid | cot | ✅ | ✅ |
| cogcanvas-minimal | ✅ | ❌ | semantic | direct | ❌ | ✅ |

---

## Retrieval Recall Analysis

Evaluate retrieval quality without LLM answer generation:

```bash
# Full system retrieval recall
python -m experiments.eval_retrieval_recall --agent cogcanvas

# Without graph expansion (for comparison)
python -m experiments.eval_retrieval_recall --agent cogcanvas-no-graph
```

---

## Question Type Analysis

```bash
# Single-hop questions (category 1)
python -m experiments.runner_locomo --agent cogcanvas --categories 1 --llm-score

# Temporal questions (category 2)
python -m experiments.runner_locomo --agent cogcanvas --categories 2 --llm-score

# Multi-hop questions (category 3)
python -m experiments.runner_locomo --agent cogcanvas --categories 3 --llm-score
```

---

## Quick Testing

For development and debugging:

```bash
# Quick test with 2 samples (fast iteration)
python -m experiments.runner_locomo --agent cogcanvas --samples 2 --llm-score --categories "1,2,3"

# Standard test with 10 samples (paper results)
python -m experiments.runner_locomo --agent cogcanvas --samples 10 --workers 3 --llm-score --categories "1,2,3"
```

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--agent` | Agent type (cogcanvas, rag, native, etc.) | required |
| `--samples N` | Limit to N conversations | all |
| `--workers N` | Parallel extraction workers | 1 |
| `--qa-parallel N` | Parallel QA workers (CogCanvas only) | 1 |
| `--llm-score` | Use LLM Judge instead of Token F1 | off |
| `--categories` | Filter by question type (1/2/3) | all |
| `--output FILE` | Save results to JSON file | auto |
| `-v` / `-vv` | Verbose output | off |

---

## Scoring Methods

| Method | Description | Used By |
|--------|-------------|---------|
| Token F1 | Strict token overlap | LoCoMo paper |
| **LLM Judge** | Binary semantic correctness | Mem0, Zep, CogCanvas |

**Recommendation**: Use `--llm-score` for fair comparison with industry systems.

---

## Output Format

Results are saved to `experiments/results/` as JSON:

```json
{
  "agent": "cogcanvas",
  "total_questions": 961,
  "overall_accuracy": 0.454,
  "by_category": {
    "single-hop": 0.309,
    "temporal": 0.277,
    "multi-hop": 0.490
  }
}
```

---

## Caching

Extraction results are cached in `experiments/cache/`. To force re-extraction:

```bash
python -m experiments.runner_locomo --agent cogcanvas --no-cache --llm-score
```

---

## Enhanced Retrieval (Experimental)

Multi-round retrieval and query expansion for improved performance.

### Single Conversation Quick Test
```bash
# Query Expansion only (BEST: 37.8% overall)
python -m experiments.runner_locomo --agent cogcanvas-expand-only --samples 1 --categories 1,2,3 --llm-score

# Multi-round retrieval (28.0% overall, +24.4pp on multi-hop)
python -m experiments.runner_locomo --agent cogcanvas-multiround --samples 1 --categories 1,2,3 --llm-score

# Multi-round + Query Expansion combo (26.8% - not recommended)
python -m experiments.runner_locomo --agent cogcanvas-multiround-expand --samples 1 --categories 1,2,3 --llm-score

# Multi-round with Query Routing
python -m experiments.runner_locomo --agent cogcanvas-multiround-routed --samples 1 --categories 1,2,3 --llm-score
```

### Full 10 Conversations
```bash
# Query Expansion (recommended)
python -m experiments.runner_locomo --agent cogcanvas-expand-only --categories 1,2,3 --llm-score

# Multi-round
python -m experiments.runner_locomo --agent cogcanvas-multiround --categories 1,2,3 --llm-score
```

### Ablation Results (Single Conversation locomo_000)

| 配置 | Overall | Single-hop | Temporal | Multi-hop |
|------|---------|------------|----------|-----------|
| **原版 CogCanvas** | 20.9% | 33.3% | 30.5% | 44.8% |
| **+ Multi-round (0.7)** | 28.0% | 12.5% | 27.0% | **69.2%** |
| **+ Multi-round (0.5)** | 28.0% | 15.6% | 29.7% | 53.8% |
| **+ Query Expansion** | **37.8%** 🏆 | 21.9% | **43.2%** | 61.5% |
| **+ 两者组合** | 26.8% | 12.5% | 29.7% | 53.8% |

**结论**: Query Expansion 单独效果最好，组合反而互相干扰。

---

## Recall-Boost Configuration (2026-01-06)

结合检索增强和答案生成改进的配置。

### 改进内容
1. **检索参数增强**: `top_k=20`, `graph_hops=4`, `reranker_candidate_k=40`
2. **Prompt 改进**: 要求直接回答、不说"不知道"、简洁

### 全量测试结果 (10 conversations, Cat 1/2/3, LLM Judge)

| 配置 | Overall | Single-hop | Temporal | Multi-hop |
|------|---------|------------|----------|-----------|
| **CogCanvas (原版)** | 28.6% | 20.9% | 30.5% | 44.8% |
| **CogCanvas (recall-boost)** | **32.4%** | **26.6%** | **32.7%** | 41.7% |
| **变化** | **+3.8pp** ✅ | +5.7pp | +2.2pp | -3.1pp |

### 运行命令
```bash
python -m experiments.runner_locomo --agent cogcanvas-recall-boost --categories 1,2,3 --llm-score
```

### 各对话结果
| 对话 | 准确率 |
|------|--------|
| locomo_000 | 33% |
| locomo_001 | 46% |
| locomo_002 | 29% |
| locomo_003 | 24% |
| locomo_004 | 32% |
| locomo_005 | 39% |
| locomo_006 | 24% |
| locomo_007 | 33% |
| locomo_008 | 30% |
| locomo_009 | 34% |

---

## Troubleshooting

### GraphRAG indexing fails
```bash
# Run with single worker to avoid race conditions
python -m experiments.runner_locomo --agent graphrag --workers 1 --llm-score
```

### API rate limits
Reduce parallel workers:
```bash
python -m experiments.runner_locomo --agent cogcanvas --workers 2 --qa-parallel 5 --llm-score
```

### Out of memory
Reduce sample size:
```bash
python -m experiments.runner_locomo --agent cogcanvas --samples 10 --llm-score
```
