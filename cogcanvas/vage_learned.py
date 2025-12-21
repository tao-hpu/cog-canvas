"""
VAGE Learned Components: Data-driven Vulnerability and Importance Estimation.

This module provides learned versions of:
1. VulnerabilityScorer: Trained classifier for compression loss prediction
2. LearnedSchemaDiscovery: Clustering-based type discovery
3. ImportanceModel: Learned importance estimation

Unlike the heuristic version in vage.py, these components are trained on
actual compression experiment data.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle


@dataclass
class LearnedVAGEConfig:
    """Configuration for learned VAGE components."""
    # Vulnerability model
    vulnerability_model_path: Optional[str] = None

    # Schema discovery
    num_clusters: int = 8  # Number of learned types
    schema_model_path: Optional[str] = None

    # Feature extraction
    use_embeddings: bool = True
    embedding_dim: int = 384


class VulnerabilityScorer:
    """
    Learned vulnerability scorer trained on compression experiments.

    Predicts P(information lost after compression) based on:
    - Position in conversation
    - Content characteristics (length, entities, numbers)
    - Object type
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'position_ratio', 'content_length_norm', 'quote_length_norm',
            'has_numbers', 'has_named_entities',
            'type_decision', 'type_key_fact', 'type_reminder',
            'type_insight', 'type_todo',
            # Extended types
            'type_person_attribute', 'type_event', 'type_relationship',
            'type_unknown',
        ]
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VulnerabilityScorer':
        """
        Train vulnerability model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 = lost after compression, 0 = retained)

        Returns:
            self
        """
        # Handle case where all labels are the same
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            print(f"Warning: All samples have same label ({unique_labels[0]})")
            print("Using position-based heuristic instead of learning")
            self.model = None
            self.is_fitted = True
            return self

        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Print feature importances
        print("Vulnerability Model Coefficients:")
        for name, coef in zip(self.feature_names[:len(self.model.coef_[0])],
                             self.model.coef_[0]):
            print(f"  {name}: {coef:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict vulnerability probability.

        Args:
            X: Feature matrix

        Returns:
            Array of P(lost) for each sample
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model is None:
            # Fallback to position-based heuristic
            # Assume first feature is position_ratio
            position = X[:, 0] if X.ndim > 1 else X
            # Higher position = lower vulnerability
            vulnerability = 1.0 - position
            return vulnerability

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]  # P(lost)

    def score_object(
        self,
        turn_id: int,
        total_turns: int,
        content: str,
        obj_type: str = 'unknown',
    ) -> float:
        """
        Score a single object's vulnerability.

        Args:
            turn_id: Which turn this appeared in
            total_turns: Total turns in conversation
            content: Text content
            obj_type: Type of object

        Returns:
            Vulnerability score in [0, 1]
        """
        import re

        # Extract features
        position_ratio = turn_id / max(total_turns, 1)
        content_length = len(content) / 100
        has_numbers = float(bool(re.search(r'\d+', content)))
        has_entities = float(self._has_entities(content))

        # One-hot for type (including extended types)
        types = ['decision', 'key_fact', 'reminder', 'insight', 'todo',
                 'person_attribute', 'event', 'relationship', 'unknown']
        type_features = [1.0 if obj_type == t else 0.0 for t in types]

        features = [
            position_ratio, content_length, content_length,  # quote_length ~ content_length
            has_numbers, has_entities,
        ] + type_features

        X = np.array([features])
        return float(self.predict_proba(X)[0])

    def _has_entities(self, text: str) -> bool:
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper() and not word.isupper():
                return True
        return False

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
            }, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']


class LearnedSchemaDiscovery:
    """
    Discover object types through clustering.

    Instead of fixed 5 types (DECISION, TODO, etc.), we cluster
    extracted objects to discover natural groupings.
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_descriptions = {}
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray, contents: List[str] = None) -> 'LearnedSchemaDiscovery':
        """
        Fit clustering model on object embeddings.

        Args:
            embeddings: Object embedding matrix (n_objects, embedding_dim)
            contents: Optional list of content strings for analysis

        Returns:
            self
        """
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        self.cluster_labels = self.kmeans.fit_predict(embeddings)
        self.is_fitted = True

        # Analyze clusters if contents provided
        if contents is not None:
            self._analyze_clusters(contents)

        return self

    def _analyze_clusters(self, contents: List[str]):
        """Analyze cluster contents to generate descriptions."""
        from collections import Counter

        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_contents = [c for c, m in zip(contents, mask) if m]

            # Find common words/patterns
            all_words = []
            for c in cluster_contents:
                words = c.lower().split()
                all_words.extend(words)

            common = Counter(all_words).most_common(5)
            self.cluster_descriptions[cluster_id] = {
                'size': sum(mask),
                'common_words': [w for w, _ in common],
                'sample': cluster_contents[0] if cluster_contents else '',
            }

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new objects.

        Args:
            embeddings: Object embedding matrix

        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.kmeans.predict(embeddings)

    def get_cluster_importance(self, cluster_id: int) -> float:
        """
        Get learned importance weight for a cluster.

        Clusters with more "decision-like" content get higher weights.
        This can be tuned based on downstream task performance.
        """
        # Default: all clusters equally important
        # Can be calibrated on evaluation data
        return 1.0 / self.n_clusters

    def print_clusters(self):
        """Print cluster analysis."""
        print(f"\n{'='*50}")
        print(f"Learned Schema: {self.n_clusters} clusters")
        print(f"{'='*50}")

        for cid, desc in self.cluster_descriptions.items():
            print(f"\nCluster {cid} (n={desc['size']}):")
            print(f"  Common words: {', '.join(desc['common_words'])}")
            print(f"  Sample: {desc['sample'][:80]}...")

    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'n_clusters': self.n_clusters,
                'cluster_descriptions': self.cluster_descriptions,
                'is_fitted': self.is_fitted,
            }, f)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.n_clusters = data['n_clusters']
            self.cluster_descriptions = data['cluster_descriptions']
            self.is_fitted = data['is_fitted']


class LearnedImportanceModel:
    """
    Learned importance estimation based on downstream task performance.

    Importance = P(this info will be queried later)

    Can be trained on:
    - Which facts were asked about in test questions
    - Which objects were retrieved for successful answers
    """

    def __init__(self):
        self.type_weights = {
            'decision': 1.0,
            'key_fact': 0.9,
            'todo': 0.8,
            'reminder': 0.85,
            'insight': 0.7,
            # Extended types for social conversations
            'person_attribute': 0.95,  # Personal info is critical
            'event': 0.9,              # Events with time are important
            'relationship': 0.85,      # Interpersonal connections
            'unknown': 0.5,
        }
        self.cluster_weights = {}  # Learned from data
        self.is_fitted = False

    def fit_from_retrieval_data(
        self,
        queries: List[str],
        retrieved_objects: List[List[Dict]],
        correct_answers: List[bool],
    ) -> 'LearnedImportanceModel':
        """
        Learn importance from retrieval success data.

        Objects that appear in successful retrievals are more important.
        """
        # Count how often each type appears in successful retrievals
        type_success = {t: 0 for t in self.type_weights}
        type_total = {t: 0 for t in self.type_weights}

        for objs, correct in zip(retrieved_objects, correct_answers):
            for obj in objs:
                obj_type = obj.get('type', 'unknown')
                type_total[obj_type] = type_total.get(obj_type, 0) + 1
                if correct:
                    type_success[obj_type] = type_success.get(obj_type, 0) + 1

        # Update weights based on success rate
        for t in self.type_weights:
            if type_total.get(t, 0) > 0:
                success_rate = type_success.get(t, 0) / type_total[t]
                # Blend with prior
                self.type_weights[t] = 0.5 * self.type_weights[t] + 0.5 * success_rate

        self.is_fitted = True
        return self

    def estimate(
        self,
        content: str,
        obj_type: str = 'unknown',
        confidence: float = 1.0,
        cluster_id: Optional[int] = None,
    ) -> float:
        """
        Estimate importance of an object.

        Args:
            content: Object content
            obj_type: Object type (decision, key_fact, etc.)
            confidence: Extraction confidence
            cluster_id: Optional learned cluster ID

        Returns:
            Importance score in [0, 1]
        """
        # Base importance from type
        base = self.type_weights.get(obj_type, 0.5)

        # Adjust by cluster if available
        if cluster_id is not None and cluster_id in self.cluster_weights:
            cluster_weight = self.cluster_weights[cluster_id]
            base = 0.7 * base + 0.3 * cluster_weight

        # Combine with confidence
        importance = base * confidence

        # Boost for specific patterns
        if any(keyword in content.lower() for keyword in ['must', 'require', 'critical', 'important']):
            importance = min(1.0, importance * 1.2)

        return importance


def train_vulnerability_model(
    data_path: str = 'experiments/data/vage_training_data.json',
) -> VulnerabilityScorer:
    """
    Train vulnerability model from prepared data.

    Args:
        data_path: Path to training data JSON

    Returns:
        Trained VulnerabilityScorer
    """
    # Load data
    with open(data_path) as f:
        data = json.load(f)

    X = np.array(data['X'])
    y = np.array(data['y'])

    print(f"Training vulnerability model on {len(y)} samples")
    print(f"Class distribution: {np.bincount(y)}")

    # Train
    scorer = VulnerabilityScorer()
    scorer.fit(X, y)

    return scorer


def discover_schema(
    objects_path: str = None,
    embeddings: np.ndarray = None,
    contents: List[str] = None,
    n_clusters: int = 8,
) -> LearnedSchemaDiscovery:
    """
    Discover object schema through clustering.

    Either provide objects_path to a JSON with embeddings,
    or provide embeddings and contents directly.

    Args:
        objects_path: Path to objects JSON with embeddings
        embeddings: Pre-computed embeddings
        contents: Object contents for analysis
        n_clusters: Number of clusters

    Returns:
        Fitted LearnedSchemaDiscovery
    """
    if embeddings is None and objects_path is not None:
        with open(objects_path) as f:
            data = json.load(f)
        embeddings = np.array([obj['embedding'] for obj in data['objects']])
        contents = [obj['content'] for obj in data['objects']]

    if embeddings is None:
        raise ValueError("Must provide either objects_path or embeddings")

    print(f"Discovering schema from {len(embeddings)} objects")
    print(f"Clustering into {n_clusters} types")

    schema = LearnedSchemaDiscovery(n_clusters=n_clusters)
    schema.fit(embeddings, contents)
    schema.print_clusters()

    return schema


# Integrated Learned VAGE
class LearnedVAGE:
    """
    Learned version of VAGE that uses trained components.

    Combines:
    - Trained VulnerabilityScorer (ρ)
    - Learned Schema Discovery
    - Importance estimation
    """

    def __init__(
        self,
        vulnerability_scorer: Optional[VulnerabilityScorer] = None,
        schema_discovery: Optional[LearnedSchemaDiscovery] = None,
        importance_model: Optional[LearnedImportanceModel] = None,
    ):
        self.vulnerability_scorer = vulnerability_scorer or VulnerabilityScorer()
        self.schema_discovery = schema_discovery
        self.importance_model = importance_model or LearnedImportanceModel()

    def prioritize(
        self,
        objects: List[Any],
        total_turns: int,
        budget_k: int = 10,
    ) -> Tuple[List[int], List[float]]:
        """
        Prioritize objects using learned components.

        Args:
            objects: List of CanvasObjects
            total_turns: Total conversation turns
            budget_k: Maximum objects to select

        Returns:
            (selected_indices, marginal_gains)
        """
        if not objects:
            return [], []

        gains = []
        for i, obj in enumerate(objects):
            # Get vulnerability (learned or heuristic)
            if self.vulnerability_scorer.is_fitted:
                rho = 1.0 - self.vulnerability_scorer.score_object(
                    turn_id=getattr(obj, 'turn_id', total_turns),
                    total_turns=total_turns,
                    content=obj.content,
                    obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
                )
            else:
                # Fallback to position heuristic
                recency = getattr(obj, 'turn_id', total_turns) / max(total_turns, 1)
                rho = 0.1 + 0.8 * recency

            vulnerability = 1.0 - rho

            # Get importance
            omega = self.importance_model.estimate(
                content=obj.content,
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
                confidence=getattr(obj, 'confidence', 1.0),
            )

            # Marginal gain
            delta = omega * vulnerability
            gains.append((i, delta))

        # Sort by gain
        gains.sort(key=lambda x: x[1], reverse=True)

        # Select top-K
        selected_indices = [g[0] for g in gains[:budget_k]]
        marginal_gains = [g[1] for i, g in enumerate(gains)]

        return selected_indices, marginal_gains


def create_learned_vage(
    vulnerability_model_path: str = None,
    schema_model_path: str = None,
    use_cluster_weights: bool = True,
) -> LearnedVAGE:
    """
    Factory function to create a LearnedVAGE with trained components.

    Args:
        vulnerability_model_path: Path to trained vulnerability scorer (.pkl)
        schema_model_path: Path to trained schema discovery model (.pkl)
        use_cluster_weights: Whether to use cluster-based importance weights

    Returns:
        Configured LearnedVAGE instance
    """
    # Load vulnerability scorer
    vulnerability_scorer = VulnerabilityScorer()
    if vulnerability_model_path and Path(vulnerability_model_path).exists():
        vulnerability_scorer.load(vulnerability_model_path)
        print(f"Loaded vulnerability model from {vulnerability_model_path}")
    else:
        print("Using default vulnerability scorer (heuristic)")

    # Load schema discovery
    schema_discovery = None
    if schema_model_path and Path(schema_model_path).exists():
        schema_discovery = LearnedSchemaDiscovery()
        schema_discovery.load(schema_model_path)
        print(f"Loaded schema model from {schema_model_path}")

    # Create importance model with cluster weights if available
    importance_model = LearnedImportanceModel()
    if use_cluster_weights and schema_discovery:
        # Set cluster weights based on vulnerability analysis
        # Cluster 0 (database/hosting) is most vulnerable, needs higher importance
        importance_model.cluster_weights = {
            0: 1.2,  # High vulnerability cluster
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 1.0,
            6: 1.0,
            7: 1.0,
        }
        importance_model.is_fitted = True

    return LearnedVAGE(
        vulnerability_scorer=vulnerability_scorer,
        schema_discovery=schema_discovery,
        importance_model=importance_model,
    )


# For backward compatibility with vage.py interface
class VAGEResult:
    """Result of VAGE extraction prioritization (compatible with vage.py)."""

    def __init__(
        self,
        selected_indices: List[int],
        marginal_gains: List[float],
        total_retained: float = 0.0,
        theoretical_optimal: float = 0.0,
    ):
        self.selected_indices = selected_indices
        self.marginal_gains = marginal_gains
        self.total_retained = total_retained
        self.theoretical_optimal = theoretical_optimal


class LearnedVAGEWrapper:
    """
    Wrapper to make LearnedVAGE compatible with the VAGE interface in canvas.py.

    This allows switching between rule-based VAGE and learned VAGE seamlessly.
    """

    def __init__(
        self,
        vulnerability_model_path: str = None,
        schema_model_path: str = None,
        default_budget_k: int = 10,
    ):
        self.learned_vage = create_learned_vage(
            vulnerability_model_path=vulnerability_model_path,
            schema_model_path=schema_model_path,
        )
        self.default_budget_k = default_budget_k

    def prioritize(
        self,
        objects: List[Any],
        total_turns: int,
        budget_k: int = None,
    ) -> VAGEResult:
        """
        Prioritize objects using learned VAGE.

        Returns VAGEResult compatible with vage.py interface.
        """
        budget_k = budget_k or self.default_budget_k

        selected_indices, marginal_gains = self.learned_vage.prioritize(
            objects=objects,
            total_turns=total_turns,
            budget_k=budget_k,
        )

        # Compute totals
        total_retained = sum(marginal_gains[i] for i in selected_indices) if selected_indices else 0
        theoretical_optimal = sum(marginal_gains)

        return VAGEResult(
            selected_indices=selected_indices,
            marginal_gains=marginal_gains,
            total_retained=total_retained,
            theoretical_optimal=theoretical_optimal,
        )


if __name__ == '__main__':
    # Demo: Train vulnerability model
    import os
    os.chdir(Path(__file__).parent.parent)

    print("Training Vulnerability Model...")
    print("=" * 50)

    try:
        scorer = train_vulnerability_model()
        print("\nModel trained successfully!")

        # Test prediction
        test_vuln = scorer.score_object(
            turn_id=3,
            total_turns=50,
            content="Use PostgreSQL for the database",
            obj_type="decision",
        )
        print(f"\nTest: Turn 3/50, 'Use PostgreSQL' -> Vulnerability: {test_vuln:.3f}")

        test_vuln2 = scorer.score_object(
            turn_id=45,
            total_turns=50,
            content="Final review needed",
            obj_type="todo",
        )
        print(f"Test: Turn 45/50, 'Final review' -> Vulnerability: {test_vuln2:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
