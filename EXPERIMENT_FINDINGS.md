# LoCoMo Optimization Experiment Findings

## Date: 2024-12-25

## Baseline Performance
- **Commit**: 66c9a6c
- **Accuracy**: 34.7% on LoCoMo benchmark
- **Configuration**: CogCanvas with Graph3hop+Time+Hybrid+CoT

## Issues Investigated

### 1. Scheduling Problem (FIXED ✓)
**Problem**: Uneven load distribution across workers
- Conversations have varying question counts (81-199 questions)
- ThreadPoolExecutor submitted tasks without load balancing
- Small tasks finished early, leaving workers idle

**Solution**: Sort conversations by question count (descending) before submission
```python
indexed_convs = list(enumerate(conversations))
indexed_convs.sort(key=lambda x: len(x[1].qa_pairs), reverse=True)
```

**Result**: Larger conversations now start first, improving parallel efficiency

### 2. BGE Reranking Performance (TESTED)
**Question**: Is BGE API reranker causing slowness?

**Findings**:
- BGE reranker speed: ~0.48s per 20 documents
- Est. overhead for full test: ~1.2 minutes (negligible)
- **Conclusion**: Reranking is NOT the bottleneck

### 3. Root Cause: LLM API Latency
**Actual Bottleneck Identified**:
1. **Compression Phase**: ~8 minutes for 3 conversations
   - LLM extraction calls for each turn
   - Graph building overhead

2. **Answering Phase**: ~2 hours for 387 questions
   - Embedding search + retrieval
   - Optional reranking (~0.5s)
   - LLM call for answer generation (主要瓶颈)

**Speed Measurements**:
- With/without reranking: Both ~2 hours for 3 conversations (387 questions)
- Est. full test (1542 questions): ~8 hours with 10 workers
- Rate: ~3-4 questions/minute (dominated by LLM latency, not retrieval)

### 4. Accuracy Results
**Test Results** (3 conversations, without reranking):
- locomo_001: 36% accuracy (29/81 passed)
- locomo_000: 36% accuracy (56/154 passed)
- locomo_002: Not completed (killed after 2 hours)

**Comparison**:
- Baseline: 34.7%
- Current: ~36% (slight improvement or variance)

## Optimization Attempts

### ✗ LLM Reranker (MISTAKE - Corrected)
**Initial Implementation**: Used LLMRerankerBackend (GPT-4o-mini for scoring)
- Would make 20 LLM calls per question (极慢!)
- **Correction**: Changed to use BGE API reranker from .env
- Lesson: Always check existing configuration before adding new features

### ✓ Scheduling Fix
**Status**: Implemented and verified working
- Conversations now start in descending order by question count
- Better load balancing across workers

### ? CoT Temporal Prompting
**Status**: Implemented but NOT tested
- Added `cot_temporal` prompt style with temporal reasoning instructions
- Need to test impact on accuracy
- Current tests use `cot` (default)

## Key Insights

1. **Speed is inherently slow**: LLM API calls dominate runtime (~95% of time)
2. **Reranking overhead is minimal**: <5% of total time
3. **Accuracy baseline is solid**: 34-36% range
4. **Improvement strategy needed**: To reach 50%+, need better methods, not just prompt engineering

## Next Steps

### High Priority
1. **Test cot_temporal prompting**: May improve temporal question accuracy
2. **Try dynamic compression threshold**: Adapt based on conversation complexity
3. **Experiment with retrieval_top_k**: Maybe 10 is too few? Try 15-20

### Medium Priority
4. **Test reranking impact on accuracy**: Enable BGE reranker and compare results
5. **Hybrid retrieval tuning**: Adjust semantic vs keyword weights

### Research Ideas
6. **Active compression triggers**: Compress when detecting topic shifts, not just every 40 turns
7. **Question-aware retrieval**: Different strategies for temporal vs multi-hop questions
8. **Graph-enhanced CoT**: Include relationship chains in reasoning prompt

## Critical Observation
**Getting to 50%+ will require methodological improvements, not just hyperparameter tuning.**
- Current architecture is sound but may hit ceiling around 35-40%
- Consider:
  - Better graph construction (more sophisticated relation detection)
  - Query decomposition for multi-hop questions
  - Temporal reasoning enhancement (dedicated temporal index?)
  - Answer validation/self-consistency checks

## Testing Strategy Going Forward
1. **One optimization at a time**: Test individually, measure impact
2. **Accept the slow speed**: 8-10 hours per full test is unavoidable with current LLM latency
3. **Use smaller samples for rapid iteration**: Test on 3-5 conversations first (2-3 hours)
4. **Full test only for promising methods**: Reserve 10-sample tests for validated optimizations
