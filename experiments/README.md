# CogCanvas Experiments

> TODO: Summarize experiment details from `paper/tasks/DOC.md` when ready.

## Quick Reference

```bash
# LoCoMo Benchmark
python -m experiments.runner_locomo --agent cogcanvas-3hop -n 10 -w 5

# MultiHop Benchmark
python -m experiments.runner_multihop --agent cogcanvas -n 50 -w 5

# Synthetic Benchmark
python -m experiments.runner --agent cogcanvas -n 50 -w 5
```

## Directory Structure

```
experiments/
├── runner.py              # Synthetic benchmark runner
├── runner_multihop.py     # MultiHop benchmark runner
├── runner_locomo.py       # LoCoMo benchmark runner
├── test_extraction_phase.py   # Extraction phase testing
├── test_answer_phase.py       # Answer phase testing
├── agents/                # Agent implementations
├── data/                  # Evaluation datasets
└── results/               # Experiment results (JSON)
```
