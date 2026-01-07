# CogCanvas

> **逐字引用的认知工件提取：LLM 长对话记忆框架**

中文版 | [English](./README.md)

[![arXiv](https://img.shields.io/badge/arXiv-2601.00821-b31b1b.svg)](https://arxiv.org/abs/2601.00821)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

CogCanvas 是一个**无需训练**的 LLM 长对话记忆框架。灵感来自团队使用白板锚定共享知识的方式，CogCanvas 提取**逐字引用的认知工件**（决策、事实、待办），并将它们组织成可查询的图结构。

**问题**：对话摘要会丢失关键细节。当被问到"我们商定的代码风格是什么？"时，摘要返回"使用类型提示"，但丢失了约束词"**所有地方**"（精确匹配 19.0% vs CogCanvas 93.0%）。

**方案**：提取带有精确来源引用的结构化工件，实现可追溯、抗幻觉的检索。

## 核心结果

### LoCoMo 基准测试（二分类 LLM 评判）

| 方法 | 整体 | 单跳 | 时序 | 多跳 |
|------|------|------|------|------|
| **CogCanvas** | **32.4%** | **26.6%** | **32.7%** | **41.7%** |
| RAG | 24.6% | 24.6% | 12.1% | 40.6% |
| GraphRAG | 10.6% | 12.8% | 3.1% | 29.2% |
| Summarization | 5.6% | 5.3% | 0.6% | 22.9% |

**核心发现**：CogCanvas 在无训练方法中达到最高整体准确率：
- **整体**: 比 RAG 高 +7.8pp (32.4% vs 24.6%)
- **时序推理**: 比 RAG 高 +20.6pp (32.7% vs 12.1%)
- **多跳问题**: 比 RAG 高 +1.1pp (41.7% vs 40.6%)
- **单跳检索**: 比 RAG 高 +2.0pp (26.6% vs 24.6%)

### 受控基准测试

| 指标 | Summarization | RAG | **CogCanvas** |
|------|---------------|-----|---------------|
| 召回率 | 19.0% | 89.5% | **97.5%** |
| 精确匹配 | 19.0% | 89.5% | **93.0%** |
| 多跳通过率 | 55.5% | 55.5% | **81.0%** |

## 快速开始

### 安装

```bash
git clone https://github.com/tao-hpu/cog-canvas.git
cd cog-canvas
pip install -e .
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1
EXTRACTOR_MODEL=gpt-4o-mini
ANSWER_MODEL=gpt-4o-mini
```

### 基本使用

```python
from cogcanvas import Canvas

# 初始化
canvas = Canvas()

# 从对话中提取
canvas.extract(
    user="我们用 PostgreSQL 作为数据库，预算是每月 500 美元。",
    assistant="好的选择，我会记录预算约束。"
)

# ... 50 轮对话后 ...

# 带图扩展的检索
results = canvas.retrieve("为什么选 PostgreSQL？预算是多少？")
# -> 返回关联工件: [DECISION: PostgreSQL] <-caused_by- [KEY_FACT: $500 预算]

# 注入到提示词
context = canvas.inject(results)
```

## 核心概念

### 工件类型

CogCanvas 提取 5 种认知工件：

| 类型 | 描述 | 示例 |
|------|------|------|
| `DECISION` | 做出的决策 | "使用 PostgreSQL 作为数据库" |
| `KEY_FACT` | 重要事实 | "预算：$500/月" |
| `TODO` | 待办事项 | "配置数据库迁移" |
| `REMINDER` | 约束条件 | "必须支持 JSONB 查询" |
| `INSIGHT` | 洞察结论 | "PostgreSQL 满足需求" |

### 逐字引用

每个工件包含 `quote` 字段，保留精确来源文本：

```json
{
  "type": "decision",
  "content": "使用 PostgreSQL 作为主数据库",
  "quote": "我们用 PostgreSQL 作为数据库",
  "turn_id": 12
}
```

这实现了：
- **可追溯性**：知道信息的精确来源
- **抗幻觉**：可对照原文验证
- **抗压缩**：保留摘要会丢失的细节

### 图增强检索

工件通过类型化关系连接：

```
[DECISION: 使用 PostgreSQL]
    ├── caused_by → [KEY_FACT: $500 预算]
    ├── caused_by → [REMINDER: 需要 JSONB 支持]
    └── leads_to → [TODO: 配置迁移]
```

检索会沿图扩展（默认 3 跳），找到简单向量搜索会遗漏的相关上下文。

## 项目结构

```
cog-canvas/
├── cogcanvas/              # 核心库
│   ├── canvas.py           # 主 Canvas 类
│   ├── models.py           # 数据模型
│   ├── graph.py            # 图操作
│   ├── temporal.py         # 时序增强
│   └── llm/                # LLM 后端
├── experiments/            # 评测代码
│   ├── runner_locomo.py    # LoCoMo 基准测试
│   └── agents/             # Agent 实现
├── web/                    # Web UI (FastAPI + Next.js)
├── EXPERIMENTS.md          # 复现指南
└── README.md
```

## 实验复现

完整复现指南见 [EXPERIMENTS.md](./EXPERIMENTS.md)。

### 快速测试

```bash
# 在 10 个样本上运行 CogCanvas
python -m experiments.runner_locomo --agent cogcanvas --samples 10 --llm-score

# 与 RAG 基线对比
python -m experiments.runner_locomo --agent rag --samples 10 --llm-score
```

### 消融实验

```bash
# 移除 BGE 重排序（影响最大：-11.5pp）
python -m experiments.runner_locomo --agent cogcanvas-no-rerank --llm-score

# 移除图扩展（-6.6pp）
python -m experiments.runner_locomo --agent cogcanvas-no-graph --llm-score
```

## Web 界面

### 启动后端（端口 3801）

```bash
cd web/backend
pip install -r requirements.txt
python main.py
```

### 启动前端（端口 3800）

```bash
cd web/frontend
pnpm install && pnpm dev
```

打开 [http://localhost:3800](http://localhost:3800)

## API 参考

### Canvas 类

```python
from cogcanvas import Canvas, ObjectType

# 使用自定义模型初始化
canvas = Canvas(
    extractor_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small"
)

# 从对话提取工件
canvas.extract(user="...", assistant="...")

# 检索相关工件
results = canvas.retrieve(
    query="为什么选择 X？",
    top_k=5,
    expand_hops=3  # 图扩展深度
)

# 按类型过滤
decisions = canvas.retrieve(query, obj_type=ObjectType.DECISION)

# 格式化用于提示词注入
context = canvas.inject(results, format="markdown")
```

## 引用

如果你在研究中使用 CogCanvas，请引用：

```bibtex
@article{an2026cogcanvas,
  title={CogCanvas: Verbatim-Grounded Artifact Extraction for Long LLM Conversations},
  author={An, Tao},
  journal={arXiv preprint arXiv:2601.00821},
  year={2026},
  url={https://arxiv.org/abs/2601.00821}
}
```

## 许可证

MIT 许可证 - 详见 [LICENSE](./LICENSE)

---

*CogCanvas：因为你的 AI 不应该忘记你们一起做出的决定。*
