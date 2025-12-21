# CogCanvas

> **Your AI's thinking whiteboard** — Paint persistent knowledge, keep your context

[中文版](./README_CN.md) | English

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

CogCanvas is a **session-level, graph-enhanced RAG system** designed to solve the "Lost in the Middle" problem in long LLM conversations. Unlike traditional summarization (which is lossy) or raw RAG (which lacks context), CogCanvas extracts structured **cognitive artifacts** (Decisions, Todos, Facts) and organizes them into a dynamic graph, enabling precise retrieval even after thousands of turns.

## Table of Contents

- [Key Results](#-key-results)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Core Concepts](#-core-concepts)
- [API Reference](#-api-reference)
- [Web UI](#-web-ui)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)

## Key Results

| Metric | Summarization | **CogCanvas** | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall (Standard)** | 86.0% | **96.5%** | **+10.5pp** |
| **Recall (Stress Test)** | 77.5% | **95.0%** | **+17.5pp** |
| **Exact Match** | 72.5% | **95.0%** | **+22.5pp** |

> "CogCanvas effectively acts as a persistent memory layer that ignores context limits."

## Quick Start

### 1. Installation

```bash
git clone https://github.com/tao-hpu/cog-canvas.git
cd cog-canvas
pip install -e .
```

### 2. Environment Configuration

**For Experiments (Python 3.10+ required):**

Some experiment baselines (e.g., GraphRAG) require Python 3.10+. We recommend using conda:

```bash
# Create and activate py310 environment
conda create -n py310 python=3.10 -y
conda activate py310

# Install llvmlite/numba via conda (avoids cmake issues)
conda install -y llvmlite numba

# Install the package
pip install -e .

# Install GraphRAG for baseline comparison
pip install graphrag
```

**Basic Setup:**

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Required: Your OpenAI-compatible API key
API_KEY=your-api-key-here
API_BASE=https://api.openai.com/v1

# Optional: Configure models
MODEL_DEFAULT=gpt-4o
```

### 3. Basic Usage

```python
from cogcanvas import Canvas

# Initialize canvas
canvas = Canvas()

# Extract cognitive objects from a dialogue turn
canvas.extract(
    user="Let's decide to use PostgreSQL for the database because of its JSONB support.",
    assistant="Great choice. I'll update the architecture diagram."
)

# ... 50 turns later ...

# Retrieve context for a new question
results = canvas.retrieve("Why did we choose Postgres?")
# -> Returns: [DECISION] Use PostgreSQL (Context: JSONB support)

# Inject into prompt
context = canvas.inject(results)
print(context)
```

## Project Structure

```
cog-canvas/
├── cogcanvas/                    # Core Python library
│   ├── __init__.py              # Package exports
│   ├── canvas.py                # Main Canvas class
│   ├── models.py                # Data models (CanvasObject, ObjectType)
│   ├── graph.py                 # Graph structure management
│   ├── embeddings.py            # Embedding backends
│   ├── resolver.py              # Relationship resolver
│   ├── scoring.py               # Confidence scoring
│   └── llm/                     # LLM backends
│       ├── base.py              # Base interface
│       ├── openai.py            # OpenAI implementation
│       └── anthropic_backend.py # Anthropic implementation
├── web/                          # Web application
│   ├── backend/                 # FastAPI backend (Port 3801)
│   │   ├── main.py              # Application entry
│   │   ├── routes/              # API routes
│   │   └── requirements.txt     # Python dependencies
│   └── frontend/                # Next.js frontend (Port 3800)
│       ├── app/                 # Next.js app router
│       ├── components/          # React components
│       └── hooks/               # Custom React hooks
├── .env.example                  # Environment template
├── pyproject.toml               # Python project config
└── README.md                    # This file
```

## Core Concepts

### Canvas Object Types

CogCanvas extracts 5 types of cognitive objects from dialogue:

| Type | Description | Example |
|------|-------------|---------|
| `DECISION` | Choices made during conversation | "Using PostgreSQL for database" |
| `TODO` | Action items and tasks | "Add error handling to login" |
| `KEY_FACT` | Important facts, numbers, names | "API rate limit: 100/min" |
| `REMINDER` | Constraints and preferences | "User prefers TypeScript" |
| `INSIGHT` | Conclusions and learnings | "Bottleneck is in DB queries" |

### Graph-Enhanced Retrieval

Objects are connected in a knowledge graph with relationships:
- **References**: Object A mentions Object B
- **Leads to**: Causal chain (Decision A → Decision B)
- **Caused by**: Reverse causal relationship

## API Reference

### Canvas Class

#### `Canvas(extractor_model, embedding_model, storage_path)`

Initialize a new canvas.

```python
from cogcanvas import Canvas

# Default (uses mock for development)
canvas = Canvas()

# With specific models
canvas = Canvas(
    extractor_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
    storage_path="./canvas.json"  # Optional: persist to file
)
```

#### `canvas.extract(user, assistant, metadata=None)`

Extract cognitive objects from a dialogue turn.

```python
result = canvas.extract(
    user="Let's use Redis for caching",
    assistant="Good idea, I'll add it to the architecture"
)

print(f"Extracted {result.count} objects")
for obj in result.objects:
    print(f"  [{obj.type.value}] {obj.content}")
```

#### `canvas.retrieve(query, top_k=5, obj_type=None, method="semantic")`

Retrieve relevant objects for a query.

```python
from cogcanvas import ObjectType

# Basic retrieval
results = canvas.retrieve("What caching solution?", top_k=3)

# Filter by type
decisions = canvas.retrieve(
    "database choices",
    obj_type=ObjectType.DECISION
)

# Include related objects (1-hop graph neighbors)
results = canvas.retrieve(
    "authentication",
    include_related=True
)
```

#### `canvas.inject(result, format="markdown", max_tokens=None)`

Format retrieved objects for prompt injection.

```python
# Markdown format (default, most readable)
context = canvas.inject(results)

# JSON format (structured)
context = canvas.inject(results, format="json")

# Compact format (minimal tokens)
context = canvas.inject(results, format="compact")

# With token budget (auto-prunes)
context = canvas.inject(results, max_tokens=500)
```

#### Utility Methods

```python
# Get all objects
all_objects = canvas.all()

# Get by type
todos = canvas.by_type(ObjectType.TODO)

# Statistics
stats = canvas.stats()
print(f"Total objects: {stats['total_objects']}")
print(f"By type: {stats['by_type']}")

# Clear canvas
canvas.clear()
```

## Web UI

### Quick Start (Two Terminals)

**Terminal 1 - Backend** (Port 3801):
```bash
cd web/backend
pip install -r requirements.txt
python main.py
```

**Terminal 2 - Frontend** (Port 3800):
```bash
cd web/frontend
pnpm install  # or: npm install
pnpm dev      # or: npm run dev
```

Open [http://localhost:3800](http://localhost:3800) to chat with CogCanvas.

### API Documentation

Once the backend is running:
- Swagger UI: [http://localhost:3801/docs](http://localhost:3801/docs)
- ReDoc: [http://localhost:3801/redoc](http://localhost:3801/redoc)

### Backend API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/canvas` | Get all canvas objects |
| GET | `/api/canvas/stats` | Get canvas statistics |
| GET | `/api/canvas/graph` | Get graph structure |
| POST | `/api/canvas/retrieve` | Retrieve relevant objects |
| POST | `/api/canvas/clear` | Clear canvas |
| POST | `/api/chat` | Chat with SSE streaming |
| POST | `/api/chat/simple` | Chat without streaming |

## Troubleshooting

### Common Issues

**1. API Key not working**
```bash
# Check your .env file
cat .env | grep API_KEY

# Verify API connectivity
curl -H "Authorization: Bearer $API_KEY" $API_BASE/models
```

**2. Port already in use**
```bash
# Find and kill process on port 3801
lsof -i :3801
kill -9 <PID>
```

**3. Module not found**
```bash
# Reinstall in development mode
pip install -e .
```

**4. CORS errors in browser**
Ensure the backend CORS configuration matches your frontend URL in `web/backend/.env`:
```
CORS_ORIGINS=http://localhost:3800
```

## Experiments

Evaluation scripts and benchmarks are in `experiments/`. See [experiments/README.md](experiments/README.md) for details.

## Roadmap

- [x] **Phase 1: MVP** - Core extraction & retrieval
- [x] **Phase 2: Core Features** - Graph linking & confidence scoring
- [x] **Phase 3: Evaluation** - Synthetic benchmarks (96.5% Recall)
- [x] **Phase 4: Web UI** - Next.js visualization dashboard
- [ ] **Phase 5: Dynamic Correction** - Handling conflicting decisions
- [ ] **Phase 6: Deployment** - PyPI release

## License

MIT License - see [LICENSE](./LICENSE) for details.

---

*CogCanvas: Because your AI shouldn't forget what you decided together.*
