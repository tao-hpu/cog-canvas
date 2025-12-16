# CogCanvas

> **AI 的思维白板** — 绘制持久知识，保持上下文

中文版 | [English](./README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

CogCanvas 是一个**会话级图增强 RAG 系统**，旨在解决 LLM 长对话中的"中间信息丢失"问题。与传统摘要（有损）或原始 RAG（缺乏上下文）不同，CogCanvas 提取结构化的**认知对象**（决策、待办、事实），并将它们组织成动态图谱，即使在数千轮对话后仍能精确检索。

## 目录

- [核心指标](#核心指标)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心概念](#核心概念)
- [API 参考](#api-参考)
- [Web 界面](#web-界面)
- [常见问题](#常见问题)
- [路线图](#路线图)

## 核心指标

| 指标 | 传统摘要 | **CogCanvas** | 提升 |
| :--- | :--- | :--- | :--- |
| **召回率（标准）** | 86.0% | **96.5%** | **+10.5pp** |
| **召回率（压力测试）** | 77.5% | **95.0%** | **+17.5pp** |
| **精确匹配** | 72.5% | **95.0%** | **+22.5pp** |

> "CogCanvas 像一个忽略上下文限制的持久记忆层。"

## 快速开始

### 1. 安装

```bash
git clone https://github.com/tao-hpu/cog-canvas.git
cd cog-canvas
pip install -e .
```

### 2. 环境配置

复制示例环境文件并填入你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 配置：

```bash
# 必填：你的 OpenAI 兼容 API 密钥
API_KEY=your-api-key-here
API_BASE=https://api.openai.com/v1

# 可选：配置模型
MODEL_DEFAULT=gpt-4o
```

### 3. 基本使用

```python
from cogcanvas import Canvas

# 初始化画布
canvas = Canvas()

# 从对话轮次中提取认知对象
canvas.extract(
    user="我们决定使用 PostgreSQL 作为数据库，因为它支持 JSONB。",
    assistant="很好的选择。我会更新架构图。"
)

# ... 50 轮对话后 ...

# 检索相关上下文
results = canvas.retrieve("我们为什么选择 Postgres？")
# -> 返回: [DECISION] 使用 PostgreSQL（上下文：JSONB 支持）

# 注入到提示词中
context = canvas.inject(results)
print(context)
```

## 项目结构

```
cog-canvas/
├── cogcanvas/                    # 核心 Python 库
│   ├── __init__.py              # 包导出
│   ├── canvas.py                # 主 Canvas 类
│   ├── models.py                # 数据模型（CanvasObject, ObjectType）
│   ├── graph.py                 # 图结构管理
│   ├── embeddings.py            # 嵌入后端
│   ├── resolver.py              # 关系解析器
│   ├── scoring.py               # 置信度评分
│   └── llm/                     # LLM 后端
│       ├── base.py              # 基础接口
│       ├── openai.py            # OpenAI 实现
│       └── anthropic_backend.py # Anthropic 实现
├── web/                          # Web 应用
│   ├── backend/                 # FastAPI 后端（端口 3801）
│   │   ├── main.py              # 应用入口
│   │   ├── routes/              # API 路由
│   │   └── requirements.txt     # Python 依赖
│   └── frontend/                # Next.js 前端（端口 3800）
│       ├── app/                 # Next.js 应用路由
│       ├── components/          # React 组件
│       └── hooks/               # 自定义 React Hooks
├── .env.example                  # 环境变量模板
├── pyproject.toml               # Python 项目配置
└── README.md                    # 英文说明
```

## 核心概念

### 画布对象类型

CogCanvas 从对话中提取 5 种认知对象：

| 类型 | 描述 | 示例 |
|------|------|------|
| `DECISION` | 对话中做出的决策 | "使用 PostgreSQL 作为数据库" |
| `TODO` | 待办事项和任务 | "给登录功能添加错误处理" |
| `KEY_FACT` | 重要事实、数字、名称 | "API 限流：100次/分钟" |
| `REMINDER` | 约束条件和偏好 | "用户偏好 TypeScript" |
| `INSIGHT` | 结论和发现 | "瓶颈在数据库查询" |

### 图增强检索

对象通过以下关系连接成知识图谱：
- **引用（References）**：对象 A 提到对象 B
- **导向（Leads to）**：因果链（决策 A → 决策 B）
- **由...引起（Caused by）**：反向因果关系

## API 参考

### Canvas 类

#### `Canvas(extractor_model, embedding_model, storage_path)`

初始化画布。

```python
from cogcanvas import Canvas

# 默认（开发环境使用 mock）
canvas = Canvas()

# 指定具体模型
canvas = Canvas(
    extractor_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    storage_path="./canvas.json"  # 可选：持久化到文件
)
```

#### `canvas.extract(user, assistant, metadata=None)`

从对话轮次中提取认知对象。

```python
result = canvas.extract(
    user="我们用 Redis 做缓存",
    assistant="好主意，我会把它加到架构中"
)

print(f"提取了 {result.count} 个对象")
for obj in result.objects:
    print(f"  [{obj.type.value}] {obj.content}")
```

#### `canvas.retrieve(query, top_k=5, obj_type=None, method="semantic")`

检索与查询相关的对象。

```python
from cogcanvas import ObjectType

# 基本检索
results = canvas.retrieve("什么缓存方案？", top_k=3)

# 按类型过滤
decisions = canvas.retrieve(
    "数据库选择",
    obj_type=ObjectType.DECISION
)

# 包含相关对象（图的 1 跳邻居）
results = canvas.retrieve(
    "认证",
    include_related=True
)
```

#### `canvas.inject(result, format="markdown", max_tokens=None)`

格式化检索结果以注入提示词。

```python
# Markdown 格式（默认，最易读）
context = canvas.inject(results)

# JSON 格式（结构化）
context = canvas.inject(results, format="json")

# 紧凑格式（最少 token）
context = canvas.inject(results, format="compact")

# 带 token 预算（自动裁剪）
context = canvas.inject(results, max_tokens=500)
```

#### 工具方法

```python
# 获取所有对象
all_objects = canvas.all()

# 按类型获取
todos = canvas.by_type(ObjectType.TODO)

# 统计信息
stats = canvas.stats()
print(f"对象总数: {stats['total_objects']}")
print(f"按类型: {stats['by_type']}")

# 清空画布
canvas.clear()
```

## Web 界面

### 快速启动（两个终端）

**终端 1 - 后端**（端口 3801）：
```bash
cd web/backend
pip install -r requirements.txt
python main.py
```

**终端 2 - 前端**（端口 3800）：
```bash
cd web/frontend
pnpm install  # 或: npm install
pnpm dev      # 或: npm run dev
```

打开 [http://localhost:3800](http://localhost:3800) 开始使用 CogCanvas 聊天。

### API 文档

后端运行后访问：
- Swagger UI: [http://localhost:3801/docs](http://localhost:3801/docs)
- ReDoc: [http://localhost:3801/redoc](http://localhost:3801/redoc)

### 后端 API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/canvas` | 获取所有画布对象 |
| GET | `/api/canvas/stats` | 获取画布统计 |
| GET | `/api/canvas/graph` | 获取图结构 |
| POST | `/api/canvas/retrieve` | 检索相关对象 |
| POST | `/api/canvas/clear` | 清空画布 |
| POST | `/api/chat` | 流式聊天（SSE） |
| POST | `/api/chat/simple` | 非流式聊天 |

## 常见问题

### 1. API 密钥无效

```bash
# 检查 .env 文件
cat .env | grep API_KEY

# 验证 API 连通性
curl -H "Authorization: Bearer $API_KEY" $API_BASE/models
```

### 2. 端口被占用

```bash
# 查找并终止占用端口 3801 的进程
lsof -i :3801
kill -9 <PID>
```

### 3. 模块未找到

```bash
# 以开发模式重新安装
pip install -e .
```

### 4. 浏览器 CORS 错误

确保后端 CORS 配置与前端 URL 匹配，在 `web/backend/.env` 中设置：
```
CORS_ORIGINS=http://localhost:3800
```

## 路线图

- [x] **阶段 1: MVP** - 核心提取与检索
- [x] **阶段 2: 核心特性** - 图链接与置信度评分
- [x] **阶段 3: 评估** - 合成基准测试（96.5% 召回率）
- [x] **阶段 4: Web UI** - Next.js 可视化界面
- [ ] **阶段 5: 动态更正** - 处理冲突决策
- [ ] **阶段 6: 发布** - PyPI 发布

## 许可证

MIT 许可证 - 详见 [LICENSE](./LICENSE)。

---

*CogCanvas：因为你的 AI 不应该忘记你们一起做出的决定。*
