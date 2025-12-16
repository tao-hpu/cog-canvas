# CogCanvas

> **Your AI's thinking whiteboard** — Paint persistent knowledge, keep your context

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Paper-Submission_Ready-green.svg)](./paper/arxiv/paper.pdf)

CogCanvas is a **session-level, graph-enhanced RAG system** designed to solve the "Lost in the Middle" problem in long LLM conversations. Unlike traditional summarization (which is lossy) or raw RAG (which lacks context), CogCanvas extracts structured **cognitive artifacts** (Decisions, Todos, Facts) and organizes them into a dynamic graph, enabling precise retrieval even after thousands of turns.

## 🏆 Key Results

| Metric | Summarization | **CogCanvas** | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall (Standard)** | 86.0% | **96.5%** | **+10.5pp** |
| **Recall (Stress Test)** | 77.5% | **95.0%** | **+17.5pp** |
| **Exact Match** | 72.5% | **95.0%** | **+22.5pp** |

> "CogCanvas effectively acts as a persistent memory layer that ignores context limits."

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/tao-hpu/cog-canvas.git
cd cog-canvas
pip install -e .
```

### 2. Basic Usage

```python
from cogcanvas import Canvas

# Initialize (uses OpenAI/Anthropic by default)
canvas = Canvas()

# Simulate a conversation turn
canvas.extract(
    user="Let's decide to use PostgreSQL for the database because of its JSONB support.",
    assistant="Great choice. I'll update the architecture diagram."
)

# ... 50 turns later ...

# Retrieve context for a new question
relevant = canvas.retrieve("Why did we choose Postgres?")
# -> Returns: [DECISION] Use PostgreSQL (Context: JSONB support)

print(canvas.inject(relevant))
```

### 3. Run Web UI Demo

We provide a beautiful Next.js interface to visualize the extraction process in real-time.

```bash
# Backend
cd web/backend
pip install -r requirements.txt
python main.py

# Frontend (in a new terminal)
cd web/frontend
pnpm install
pnpm dev
```
Open [http://localhost:3000](http://localhost:3000) to chat with CogCanvas.

## 🧪 Reproducing Experiments

All experiments from our paper are reproducible.

### 1. Synthetic Benchmarks (Table 2 & 3)
Run the controlled information retention test:
```bash
python experiments/runner.py --turns 50 --n 10
```

### 2. Real-world Case Study (Figure 4)
Reproduce the Next.js RFC analysis:
```bash
python experiments/run_case_study.py
```

## 🧠 Core Concepts

**Canvas Object Types**:
- `DECISION`: Choices made ("Using PostgreSQL")
- `TODO`: Action items ("Add error handling")
- `KEY_FACT`: Important facts ("API limit: 100/min")
- `REMINDER`: Constraints ("User prefers concise code")
- `INSIGHT`: Conclusions ("Bottleneck is DB queries")

## 🗺️ Roadmap

- [x] **Phase 1: MVP** - Core extraction & retrieval
- [x] **Phase 2: Core Features** - Graph linking & confidence scoring
- [x] **Phase 3: Evaluation** - Synthetic benchmarks (96.5% Recall)
- [x] **Phase 4: Web UI** - Next.js visualization dashboard
- [x] **Phase 5: Paper** - Draft complete
- [x] **Phase 6: Real-world Qual** - Case study on GitHub RFC
- [ ] **Phase 7: Dynamic Correction** - Handling conflicting decisions
- [ ] **Phase 8: Deployment** - PyPI release

## 📚 Documentation

- [Task List](./tasks/TODO.md)
- [Design Doc](./tasks/DOC.md)
- [Paper Draft](./paper/arxiv/paper.pdf)

## License

MIT License - see [LICENSE](./LICENSE) for details.

---

*CogCanvas: Because your AI shouldn't forget what you decided together.*