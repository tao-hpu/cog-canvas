
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from cogcanvas.canvas import Canvas
from cogcanvas.models import CanvasObject, ObjectType
from cogcanvas.embeddings import EmbeddingBackend
from typing import List

# Mock Embedding Backend to spy on inputs
class SpyEmbeddingBackend(EmbeddingBackend):
    def __init__(self):
        pass

    @property
    def dimension(self) -> int:
        return 10

    def embed(self, text: str) -> List[float]:
        return [0.0] * 10

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        print("\n[SPY] Embedding batch received:")
        for i, text in enumerate(texts):
            print(f"  Item {i}: {text[:100]}...")  # Print first 100 chars
            if "Original Quote" in text:
                print("    -> SUCCESS: Quote found in embedding text!")
            else:
                print("    -> FAILURE: Quote NOT found in embedding text!")
        return [[0.0] * 10] * len(texts)

# Mock LLM Backend just to return an object
class MockLLMBackend:
    def extract_objects(self, *args, **kwargs):
        # Return a dummy object with a quote
        return [
            CanvasObject(
                type=ObjectType.KEY_FACT,
                content="User likes pizza",
                quote="Original Quote: I really love pizza!",
                confidence=1.0,
                id="test_obj_1"
            )
        ]

def main():
    print("Initializing Canvas with SpyBackend...")
    canvas = Canvas(extractor_model="mock", embedding_model="mock")
    
    # Inject our spy
    canvas._embedding_backend = SpyEmbeddingBackend()
    canvas._backend = MockLLMBackend()
    
    print("Running extraction...")
    canvas.extract(user="I really love pizza!", assistant="Me too.")

if __name__ == "__main__":
    main()
