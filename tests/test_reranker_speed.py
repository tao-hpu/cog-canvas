"""Test reranker speed."""
import time
import os
from dotenv import load_dotenv
from cogcanvas.reranker import Reranker

# Load environment variables
load_dotenv()

# Prepare test data
query = "When did Caroline go to the LGBTQ support group?"

documents = [
    "Caroline attended a transgender support group meeting on May 7, 2024.",
    "Melanie went to the museum with her kids last weekend.",
    "The budget for the project is $500 and we need to choose cheap hosting.",
    "John decided to use AWS for the deployment because it's affordable.",
    "Caroline is a counselor who specializes in LGBTQ+ youth support.",
    "The pottery class starts at 3pm on Tuesdays.",
    "Melanie painted a beautiful sunrise last month.",
    "Caroline gave a speech at a pride parade in June 2024.",
    "The adoption process requires several documents and background checks.",
    "Melanie's daughter's birthday is coming up next month.",
    "Caroline met up with her mentor at a coffee shop downtown.",
    "The LGBTQ center hosts weekly support groups on Wednesdays.",
    "Caroline has been working as a counselor for 4 years.",
    "Melanie recently bought some art supplies for her painting hobby.",
    "The support group discusses various topics related to identity and mental health.",
    "Caroline joined a mentorship program to help young transgender individuals.",
    "Melanie went camping with her family in July.",
    "Caroline's 18th birthday was about 6 years ago.",
    "The youth center is planning a series of workshops this fall.",
    "Caroline is passionate about creating safe spaces for LGBTQ+ youth.",
]

print("Testing BGE API Reranker speed...")
print(f"Query: {query}")
print(f"Documents: {len(documents)}")

# Test API reranker
try:
    reranker = Reranker(use_mock=False)
    print("\n=== API Reranker (BGE) ===")

    # Warm up
    print("Warming up...")
    reranker.rerank(query, documents[:5], top_k=3)

    # Test multiple times
    times = []
    for i in range(5):
        start = time.time()
        results = reranker.rerank(query, documents, top_k=10)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"\n  Average: {avg_time:.3f}s per rerank")
    print(f"  Top result: {documents[results[0][0]][:80]}... (score: {results[0][1]:.3f})")

    # Calculate impact on full test
    total_questions = 1542
    estimated_rerank_time = total_questions * avg_time
    print(f"\n  Estimated total reranking time for {total_questions} questions:")
    print(f"    {estimated_rerank_time:.1f}s = {estimated_rerank_time/60:.1f} minutes")
    print(f"    With 10 workers: {estimated_rerank_time/600:.1f} minutes")

except Exception as e:
    print(f"API reranker failed: {e}")
    print("Falling back to mock reranker test...")

    # Test mock reranker as fallback
    reranker = Reranker(use_mock=True)
    print("\n=== Mock Reranker ===")

    times = []
    for i in range(5):
        start = time.time()
        results = reranker.rerank(query, documents, top_k=10)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"\n  Average: {avg_time:.3f}s per rerank")
    print(f"  Top result: {documents[results[0][0]][:80]}... (score: {results[0][1]:.3f})")
