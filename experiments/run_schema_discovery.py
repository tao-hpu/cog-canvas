"""
Run schema discovery on planted facts to demonstrate learned types.

This script:
1. Loads planted facts from eval datasets
2. Computes embeddings for each fact
3. Clusters to discover natural groupings
4. Compares discovered types vs fixed 5 types
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict
import os
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogcanvas.vage_learned import LearnedSchemaDiscovery


def load_all_facts(data_dir: str = 'experiments/data') -> List[Dict]:
    """Load all planted facts from datasets."""
    facts = []

    datasets = ['eval_set.json', 'eval_set_hard.json', 'eval_set_long.json']

    for ds in datasets:
        path = Path(data_dir) / ds
        if not path.exists():
            continue

        with open(path) as f:
            data = json.load(f)

        for conv in data['conversations']:
            for fact in conv.get('planted_facts', []):
                facts.append({
                    'id': fact['id'],
                    'content': fact['content'],
                    'quote': fact['quote'],
                    'type': fact['fact_type'],
                    'difficulty': fact.get('difficulty', 'unknown'),
                    'turn_id': fact['turn_id'],
                    'ground_truth': fact.get('ground_truth', ''),
                })

    return facts


def compute_embeddings_simple(texts: List[str]) -> np.ndarray:
    """
    Compute simple TF-IDF based embeddings as fallback.

    For production, use sentence-transformers or API embeddings.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)

    # Reduce to 50 dimensions
    svd = TruncatedSVD(n_components=min(50, tfidf.shape[1] - 1))
    embeddings = svd.fit_transform(tfidf)

    return embeddings


def compute_embeddings_api(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings using API (if available).
    """
    try:
        from cogcanvas.embeddings import APIEmbeddingBackend

        api_key = os.environ.get('EMBEDDING_API_KEY') or os.environ.get('OPENAI_API_KEY')
        api_base = os.environ.get('EMBEDDING_API_BASE') or os.environ.get('OPENAI_API_BASE')

        if not api_key:
            print("No API key found, using TF-IDF fallback")
            return None

        backend = APIEmbeddingBackend(
            model=os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3'),
            api_key=api_key,
            api_base=api_base,
        )

        embeddings = backend.embed_batch(texts)
        return np.array(embeddings)

    except Exception as e:
        print(f"API embedding failed: {e}")
        return None


def analyze_type_distribution(facts: List[Dict]):
    """Analyze distribution of fixed types."""
    type_counts = Counter(f['type'] for f in facts)

    print("\n" + "=" * 50)
    print("Fixed Type Distribution")
    print("=" * 50)

    for t, count in type_counts.most_common():
        pct = count / len(facts) * 100
        print(f"  {t:15s}: {count:4d} ({pct:5.1f}%)")


def compare_clusters_to_types(
    facts: List[Dict],
    cluster_labels: np.ndarray,
    n_clusters: int,
):
    """Compare discovered clusters to fixed types."""
    print("\n" + "=" * 50)
    print("Cluster vs Fixed Type Mapping")
    print("=" * 50)

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_facts = [f for f, m in zip(facts, mask) if m]

        if not cluster_facts:
            continue

        # Type distribution in this cluster
        type_dist = Counter(f['type'] for f in cluster_facts)

        print(f"\nCluster {cluster_id} (n={len(cluster_facts)}):")
        for t, count in type_dist.most_common(3):
            pct = count / len(cluster_facts) * 100
            print(f"    {t}: {count} ({pct:.0f}%)")

        # Sample contents
        print("  Samples:")
        for f in cluster_facts[:3]:
            print(f"    - {f['content'][:60]}...")


def main():
    print("Schema Discovery from Planted Facts")
    print("=" * 50)

    # Load facts
    facts = load_all_facts()
    print(f"Loaded {len(facts)} facts")

    # Analyze fixed types
    analyze_type_distribution(facts)

    # Prepare texts for embedding
    texts = [f"{f['content']} {f['quote'][:100]}" for f in facts]

    # Try API embeddings first, fallback to TF-IDF
    print("\nComputing embeddings...")
    embeddings = compute_embeddings_api(texts)

    if embeddings is None:
        print("Using TF-IDF embeddings")
        embeddings = compute_embeddings_simple(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # Discover schema
    print("\n" + "=" * 50)
    print("Discovering Schema (Clustering)")
    print("=" * 50)

    # Try different cluster counts
    for n_clusters in [5, 8, 10]:
        print(f"\n--- {n_clusters} Clusters ---")

        schema = LearnedSchemaDiscovery(n_clusters=n_clusters)
        contents = [f['content'] for f in facts]
        schema.fit(embeddings, contents)

        # Compare to fixed types
        compare_clusters_to_types(facts, schema.cluster_labels, n_clusters)

    # Save schema model
    print("\n" + "=" * 50)
    print("Saving best model (8 clusters)")

    schema_8 = LearnedSchemaDiscovery(n_clusters=8)
    schema_8.fit(embeddings, [f['content'] for f in facts])

    output_dir = Path('experiments/data')
    schema_8.save(str(output_dir / 'learned_schema.pkl'))
    print(f"Saved to {output_dir / 'learned_schema.pkl'}")

    # Save embeddings for future use
    np.save(output_dir / 'fact_embeddings.npy', embeddings)
    print(f"Saved embeddings to {output_dir / 'fact_embeddings.npy'}")

    # Save facts with cluster labels
    for i, f in enumerate(facts):
        f['cluster_id'] = int(schema_8.cluster_labels[i])

    with open(output_dir / 'facts_with_clusters.json', 'w') as f:
        json.dump({'facts': facts}, f, indent=2)
    print(f"Saved facts with clusters to {output_dir / 'facts_with_clusters.json'}")


if __name__ == '__main__':
    main()
