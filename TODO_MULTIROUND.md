# CogCanvas Multi-Round Retrieval 升级计划

> 目标：从 28.6% → 40-55% (LoCoMo)
> 预计工作量：1 晚上（6-8小时）
> 核心改动：`cogcanvas_agent.py` 的 `answer_question` 方法

---

## 核心原理

当前问题：**单轮检索，找不到就没了**

```
用户问题 → 检索一次 → 返回 top-k → 生成答案
```

目标架构：**多轮迭代检索**

```
用户问题 → 检索第1轮 → 信息够吗？
                         ↓ 不够
         生成补充查询 → 检索第2轮 → 信息够吗？
                         ↓ 不够
         生成补充查询 → 检索第3轮 → 生成答案
```

---

## TODO 清单

### Phase 1: Multi-Round Retrieval (核心，+10-15pp)

**文件**: `experiments/agents/cogcanvas_agent.py`

- [ ] **1.1 添加 multi-round 参数**
  ```python
  def __init__(self, ...
      max_retrieval_rounds: int = 3,  # 最大检索轮数
      confidence_threshold: float = 0.7,  # 置信度阈值
      use_multi_round: bool = True,  # 开关
  ```

- [ ] **1.2 实现 `_check_answer_confidence` 方法**
  ```python
  def _check_answer_confidence(self, question: str, context: str, answer: str) -> float:
      """
      让 LLM 判断当前上下文是否足够回答问题
      返回 0-1 的置信度分数
      """
      prompt = f"""
      Question: {question}
      Context: {context}
      Draft Answer: {answer}

      Rate confidence (0-1) that this answer is correct and complete.
      If context is missing key information, return low score.
      Return ONLY a number between 0 and 1.
      """
      # 调用 LLM，解析返回的数字
  ```

- [ ] **1.3 实现 `_generate_followup_query` 方法**
  ```python
  def _generate_followup_query(self, question: str, context: str, answer: str) -> str:
      """
      根据当前上下文和答案，生成补充查询
      """
      prompt = f"""
      Original question: {question}
      Current context: {context}
      Current answer attempt: {answer}

      What additional information is needed to answer this question completely?
      Generate a search query to find the missing information.
      Return ONLY the search query, nothing else.
      """
  ```

- [ ] **1.4 重构 `answer_question` 方法**
  ```python
  def answer_question(self, question: str, verbose: int = 0) -> AgentResponse:
      if not self.use_multi_round:
          return self._answer_question_single(question, verbose)  # 原逻辑

      all_objects = []
      seen_ids = set()
      current_query = question

      for round_num in range(self.max_retrieval_rounds):
          # 1. 检索
          retrieval_result = self._retrieve_round(current_query, exclude_ids=seen_ids)

          # 2. 合并结果（去重）
          for obj in retrieval_result.objects:
              if obj.id not in seen_ids:
                  all_objects.append(obj)
                  seen_ids.add(obj.id)

          # 3. 构建上下文
          context = self._build_context(all_objects)

          # 4. 尝试回答
          draft_answer = self._generate_answer(question, context)

          # 5. 检查置信度
          confidence = self._check_answer_confidence(question, context, draft_answer)

          if confidence >= self.confidence_threshold:
              break  # 信息足够，退出循环

          # 6. 生成补充查询
          current_query = self._generate_followup_query(question, context, draft_answer)

      return AgentResponse(answer=draft_answer, ...)
  ```

---

### Phase 2: Query Expansion (补充，+5-10pp)

**文件**: `experiments/agents/cogcanvas_agent.py`

你已经有 `use_query_expansion` 参数，但当前实现较简单。

- [ ] **2.1 增强 `_retrieve_with_expansion` 方法**
  ```python
  def _expand_query(self, question: str) -> List[str]:
      """
      将原始问题扩展为多个子查询（Perplexity 风格）
      """
      prompt = f"""
      Original question: {question}

      Generate 3 different search queries that would help answer this question.
      Each query should focus on different aspects:
      1. Direct factual query
      2. Temporal/contextual query
      3. Related entities query

      Return as JSON: ["query1", "query2", "query3"]
      """
  ```

- [ ] **2.2 合并多查询结果**
  ```python
  def _retrieve_with_expansion(self, question: str, top_k: int) -> RetrievalResult:
      queries = [question] + self._expand_query(question)

      all_objects = []
      all_scores = {}

      for q in queries:
          result = self._canvas.retrieve(query=q, top_k=top_k)
          for obj, score in zip(result.objects, result.scores):
              if obj.id in all_scores:
                  all_scores[obj.id] = max(all_scores[obj.id], score)  # 取最高分
              else:
                  all_scores[obj.id] = score
                  all_objects.append(obj)

      # 按分数排序
      sorted_objects = sorted(all_objects, key=lambda x: all_scores[x.id], reverse=True)
      return RetrievalResult(objects=sorted_objects[:top_k], ...)
  ```

---

### Phase 3: 测试与评估

- [ ] **3.1 添加 ablation 配置**
  ```python
  # runner_locomo.py 中添加
  AGENT_CONFIGS = {
      "cogcanvas-multiround": {
          "use_multi_round": True,
          "max_retrieval_rounds": 3,
          "confidence_threshold": 0.7,
      },
      "cogcanvas-multiround-expand": {
          "use_multi_round": True,
          "use_query_expansion": True,
          "max_retrieval_rounds": 3,
      },
  }
  ```

- [ ] **3.2 运行 LoCoMo 评估**
  ```bash
  # 测试单个对话
  python -m experiments.runner_locomo \
      --agent cogcanvas-multiround \
      --conv-ids locomo_000 \
      --verbose 2

  # 全量评估
  python -m experiments.runner_locomo \
      --agent cogcanvas-multiround \
      --verbose 1
  ```

- [ ] **3.3 对比结果**
  | 配置 | Overall | Temporal | Multi-hop |
  |------|---------|----------|-----------|
  | CogCanvas (当前) | 28.6% | 30.5% | 44.8% |
  | + Multi-round | ? | ? | ? |
  | + Multi-round + Expansion | ? | ? | ? |

---

## 关键代码位置

| 文件 | 关键方法 | 说明 |
|------|----------|------|
| `cogcanvas_agent.py:705` | `answer_question` | **主要修改点** |
| `cogcanvas_agent.py:720` | `_retrieve_with_expansion` | 已有，需增强 |
| `canvas.py:40` | `retrieve` | 底层检索，不需要改 |

---

## 风险与备选

### 风险
1. **LLM 调用增加 2-3x**：每轮检索需要额外调用 confidence check + query generation
2. **延迟增加**：从 ~2s/question → ~5-8s/question
3. **成本增加**：API 费用翻倍

### 备选方案（如果时间不够）
只做 **Phase 2 Query Expansion**，不做 multi-round：
- 工作量：2-3 小时
- 预期收益：+5-8pp
- 风险低，改动小

---

## 快速开始

```bash
cd /Users/TaoTao/Desktop/Learn/硕士阶段/cog-canvas-all/cog-canvas

# 1. 编辑 cogcanvas_agent.py，添加 multi-round 逻辑

# 2. 测试单个对话
python -m experiments.runner_locomo \
    --agent cogcanvas \
    --conv-ids locomo_000 \
    --verbose 2

# 3. 验证改动生效后，全量测试
python -m experiments.runner_locomo --agent cogcanvas --verbose 1
```

---

## 参考：EverMemOS 的做法

根据 [EverMemOS GitHub](https://github.com/EverMind-AI/EverMemOS)：

1. **Agentic multi-round recall** - 生成补充查询，多轮迭代
2. **Hybrid retrieval** - BM25 + Vector + RRF fusion（你已经有 hybrid）
3. **LLM-guided contextual reasoning** - 最后一步让 LLM 整合（你已经有 CoT）

**你缺的就是 multi-round recall**，这是最大差距。
