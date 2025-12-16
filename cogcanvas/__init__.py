"""
CogCanvas: Compression-resistant cognitive objects for long LLM conversations.

Your AI's thinking whiteboard - paint persistent knowledge, keep your context.
"""

from cogcanvas.models import CanvasObject, ObjectType
from cogcanvas.canvas import Canvas
from cogcanvas.graph import CanvasGraph

__version__ = "0.1.0"
__all__ = ["Canvas", "CanvasObject", "ObjectType", "CanvasGraph"]
