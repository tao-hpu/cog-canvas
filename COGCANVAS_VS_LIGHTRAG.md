# CogCanvas vs LightRAG Architecture Comparison

## Core Philosophy

### CogCanvas (白板理论)
**本质**: 对话式长记忆管理系统
- **核心价值**: 滚动压缩 + 时间追踪 + 增量图更新
- **图 RAG 定位**: 可替换的底层组件
- **设计目标**: 管理长对话中的认知状态演化

### LightRAG
**本质**: 通用文档知识图谱系统
- **核心价值**: 实体关系网络 + 双层检索
- **设计目标**: 从静态文档构建知识图谱

## 实体/关系抽取对比

| 维度 | CogCanvas | LightRAG |
|------|-----------|----------|
| **抽取单位** | 对话轮次（turn-by-turn） | 整个文档（document-level） |
| **对象类型** | 8种认知对象（decision, todo, key_fact, person_attribute, event, relationship, reminder, insight） | 通用实体 + 关系 |
| **时间处理** | 显式时间追踪（turn_id, session_datetime, time_expression） | 弱时间感知 |
| **增量更新** | 滚动压缩（每40轮压缩，保留5轮） | 静态索引（一次性构建） |
| **提取方式** | 1次 LLM 调用/轮 + regex 时间回退 | 迭代细化抽取（entity_extract_max_gleaning） |
| **去重策略** | 基于最近10个对象上下文 | 全局去重 |

## 图结构对比

| 维度 | CogCanvas | LightRAG |
|------|-----------|----------|
| **关系类型** | 3种固定关系（references, leads_to, caused_by） | 开放式关系（LLM 自由抽取） |
| **图拓扑** | 邻接表 + 反向索引 | 向量化图（embeddings + graph） |
| **检索方式** | 3-hop 扩展 + BFS 遍历 | 双层检索（local 实体邻域 + global 关系模式） |
| **向量集成** | 对象内容向量化（分离式） | 统一向量 + 图管理（融合式） |

## 检索策略对比

| 维度 | CogCanvas | LightRAG |
|------|-----------|----------|
| **粗检索** | 向量相似度（top-k chunks） | 双层检索（实体邻域 + 全局关系） |
| **重排序** | BGE reranker（可选） | 可选 reranker |
| **上下文构建** | Graph3hop 扩展 + 时间窗口 | Token budget 控制（max_entity_tokens, max_relation_tokens） |
| **混合检索** | Semantic + Keyword（Hybrid） | Semantic + Graph traversal |

## 核心差异总结

### CogCanvas 的独特性
1. **对话中心设计**: 逐轮增量更新，非静态文档
2. **时间感知强**: 显式时间追踪，支持"在X之后发生了什么"类问题
3. **认知对象建模**: 区分 decision/todo/insight 等高层语义
4. **滚动压缩策略**: 长期记忆 + 短期上下文

### LightRAG 的优势
1. **全局实体网络**: 更强的跨文档关系推理
2. **双层检索**: 结合局部上下文 + 全局模式
3. **向量图融合**: 统一的 embedding + graph 管理
4. **迭代抽取**: 更精确的实体识别（通过 gleaning 迭代）

## 集成可行性分析

### 方案1: 用 LightRAG 替换 CogCanvas 的图构建
**可行性**: ⚠️ 困难
- LightRAG 设计为**静态文档索引**，不支持滚动更新
- 每次压缩需要重建整个图（成本高）
- 时间追踪会丢失（LightRAG 弱时间感知）

### 方案2: 借鉴 LightRAG 的双层检索
**可行性**: ✅ 可行
- 保留 CogCanvas 的增量图构建
- 增强检索策略：
  - Local 检索：当前 3-hop 扩展
  - Global 检索：跨对话的关系模式（如"所有 decision 导致的 event"）
- 统一 token budget 管理

### 方案3: 混合架构
**可行性**: ✅ 推荐
- **短期记忆**: CogCanvas 滚动压缩（对话中心）
- **长期知识**: LightRAG 索引压缩后的摘要（文档中心）
- **两层检索**:
  1. 短期：检索最近 N 轮对话 + 3-hop 扩展
  2. 长期：检索历史压缩摘要的全局知识图谱

## 性能影响预测

### 如果完全替换为 LightRAG
- ❌ 失去增量更新能力（每次压缩重建图）
- ❌ 失去时间追踪精度
- ✅ 提升实体关系质量（迭代抽取）
- ✅ 提升跨对话推理能力

### 如果借鉴双层检索
- ✅ 保留 CogCanvas 核心优势
- ✅ 增强全局推理（global retrieval）
- ⚠️ 增加检索复杂度（轻微）

### 如果混合架构
- ✅ 兼具短期精准 + 长期推理
- ✅ 分层管理复杂度
- ⚠️ 系统架构变复杂

## 回答你的核心问题

### Q1: 简单图 RAG 能打过 LightRAG 吗？
**A**: 不能，如果单纯比拼图构建质量：
- LightRAG 的迭代抽取 > CogCanvas 的单次提取
- LightRAG 的全局关系网络 > CogCanvas 的局部图

### Q2: 本质是白板理论，用什么图 RAG 无所谓？
**A**: 部分正确 ✅
- **白板理论的核心**：滚动压缩 + 时间追踪 + 增量更新
- **图 RAG 是可替换组件**：但要满足增量更新需求
- **问题**：LightRAG 不是为增量设计，直接替换会失去核心优势

### Q3: 能在 LightRAG 之上做吗？
**A**: 混合架构可行 ✅
- **短期层**: CogCanvas（对话增量）
- **长期层**: LightRAG（知识沉淀）
- **检索策略**: 双层检索（recent turns + global knowledge）

## 推荐方案

### 近期（快速提升）
1. **借鉴 LightRAG 的迭代抽取**：
   - 在 CogCanvas 的 `extract_objects` 中增加 gleaning 迭代
   - Prompt: "Review previous extraction, any missed entities?"

2. **增强双层检索**：
   - Local: 当前 3-hop（已有）
   - Global: 跨对话的关系模式查询（新增）

### 中期（架构优化）
3. **统一向量 + 图管理**：
   - 参考 LightRAG 的 token budget 控制
   - 避免上下文溢出

### 长期（研究方向）
4. **混合架构**：
   - 对话层用 CogCanvas
   - 知识层用 LightRAG 索引历史摘要
   - 实现真正的长期记忆

## 关键洞察

**你的系统不是在和 LightRAG 竞争**，而是解决不同问题：
- **LightRAG**: 静态文档的知识图谱
- **CogCanvas**: 动态对话的记忆管理

**正确的竞争对手**: Letta (MemGPT)、Cognition Graph 等**对话记忆系统**

**提分策略**: 不是替换图 RAG，而是：
1. 提升实体关系抽取质量（借鉴迭代抽取）
2. 增强检索策略（双层检索 + 时间推理）
3. 优化压缩策略（动态阈值 + 主动触发）
4. 强化 CoT 推理（图增强推理链）

## Sources
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [LightRAG Official Site](https://lightrag.github.io/)
- [Under the Covers with LightRAG](https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/)
- [RAG Upgrade for AI Development 2026](https://medium.com/@DevBoostLab/graphrag-biggest-upgrade-ai-development-2026-33366891525d)
