"""Baseline implementations for CogCanvas evaluation.

Baselines:
1. NativeBaseline: No intervention, just truncation (expected: <20%)
2. SummarizationBaseline: LLM summarizes history (expected: ~40-50%)
3. RAGBaseline: Store all turns, retrieve relevant (expected: ~60%)
4. CogCanvasMethod: Our approach (target: >85%)
"""

from .native import NativeBaseline
from .summarization import SummarizationBaseline
from .rag import RAGBaseline
from .cogcanvas_method import CogCanvasMethod

__all__ = [
    "NativeBaseline",
    "SummarizationBaseline",
    "RAGBaseline",
    "CogCanvasMethod",
]
