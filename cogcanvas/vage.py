"""
VAGE: Vulnerability-Aware Greedy Extraction

Theoretical Framework for Optimal Information Retention under Compression.

Core Insight:
    Extract priority = Importance × Vulnerability
    where Vulnerability = 1 - Retention_Probability

This module provides:
1. RetentionPredictor: Estimates P(retained after compression)
2. ImportanceEstimator: Estimates information importance
3. VAGE algorithm: Greedy selection with provable optimality
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .graph import CanvasGraph


@dataclass
class VAGEConfig:
    """Configuration for VAGE algorithm."""

    # Retention probability parameters (linear model: rho = alpha + beta * recency)
    retention_alpha: float = 0.1   # Base retention (early turns)
    retention_beta: float = 0.8    # Recency boost coefficient

    # Feature weights for retention prediction
    use_position_feature: bool = True
    use_length_feature: bool = True
    use_entity_feature: bool = True

    # Extraction budget
    default_budget_k: int = 10

    # Importance source
    importance_source: str = "confidence"  # "confidence" or "learned"


class RetentionPredictor:
    """
    Predicts the probability that information will be retained after compression.

    P(retain) is higher for:
    - Recent turns (recency bias in summarization)
    - Content with numbers/entities (more memorable)
    - Shorter, focused statements

    Model: rho(s) = alpha + beta * recency + gamma * features
    """

    def __init__(self, config: Optional[VAGEConfig] = None):
        self.config = config or VAGEConfig()

        # Learned weights (can be fitted from compression experiments)
        self.weights = {
            "recency": self.config.retention_beta,
            "has_numbers": 0.1,      # Numbers are more likely retained
            "has_entities": 0.1,     # Named entities are memorable
            "length_penalty": -0.05, # Very long content may be summarized
        }

    def predict(
        self,
        turn_id: int,
        total_turns: int,
        content: str,
        obj_type: Optional[str] = None,
    ) -> float:
        """
        Predict retention probability for a piece of information.

        Args:
            turn_id: Which turn this info appeared in
            total_turns: Total turns in conversation so far
            content: The text content
            obj_type: Type of object (decision, fact, etc.)

        Returns:
            rho in [0, 1]: Probability of being retained after compression
        """
        # Base: position-based retention
        if total_turns > 0:
            recency = turn_id / total_turns
        else:
            recency = 1.0

        rho = self.config.retention_alpha + self.weights["recency"] * recency

        # Feature adjustments
        if self.config.use_entity_feature:
            if self._has_numbers(content):
                rho += self.weights["has_numbers"]
            if self._has_named_entities(content):
                rho += self.weights["has_entities"]

        if self.config.use_length_feature:
            # Penalize very long content (> 200 chars)
            if len(content) > 200:
                rho += self.weights["length_penalty"]

        # Clamp to [0, 1]
        return max(0.0, min(1.0, rho))

    def _has_numbers(self, text: str) -> bool:
        """Check if text contains numbers."""
        return any(c.isdigit() for c in text)

    def _has_named_entities(self, text: str) -> bool:
        """Simple heuristic: capitalized words that aren't sentence starters."""
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper():
                return True
        return False

    def fit(self, training_data: List[Tuple[dict, bool]]) -> None:
        """
        Fit retention model from compression experiment data.

        Args:
            training_data: List of (features_dict, was_retained) pairs

        This can be extended to use logistic regression or other models.
        For now, we use the default heuristic weights.
        """
        # TODO: Implement actual fitting from data
        # For now, use default weights
        pass


class ImportanceEstimator:
    """
    Estimates the importance of extracted information.

    Can use:
    1. Existing confidence scores (default)
    2. Learned importance model
    3. Heuristic rules
    """

    def __init__(self, source: str = "confidence"):
        """
        Args:
            source: "confidence" to use existing scores, "learned" for trained model
        """
        self.source = source

    def estimate(
        self,
        content: str,
        confidence: float,
        obj_type: Optional[str] = None,
        context: Optional[str] = None,
    ) -> float:
        """
        Estimate importance of information.

        Args:
            content: The text content
            confidence: Existing extraction confidence
            obj_type: Type of object
            context: Additional context

        Returns:
            omega in [0, 1]: Importance score
        """
        if self.source == "confidence":
            # Directly use confidence score
            return confidence

        elif self.source == "heuristic":
            # Type-based importance
            type_weights = {
                "decision": 1.0,    # Decisions are critical
                "key_fact": 0.9,    # Facts with numbers are important
                "todo": 0.8,        # Action items matter
                "reminder": 0.7,    # Constraints/preferences
                "insight": 0.6,     # Nice to have
                # Extended types for social conversations
                "person_attribute": 0.95,  # Personal info is critical
                "event": 0.9,              # Events with time are important
                "relationship": 0.85,      # Interpersonal connections
            }
            base = type_weights.get(obj_type, 0.5)
            return base * confidence

        else:  # learned
            # TODO: Implement learned importance model
            return confidence


@dataclass
class ObjectScore:
    """Score components for a single object."""
    omega: float  # importance
    rho: float    # retention probability
    delta: float  # marginal gain = omega * (1 - rho)


@dataclass
class VAGEResult:
    """Result of VAGE extraction prioritization."""

    selected_indices: List[int]       # Indices of selected objects
    marginal_gains: List[float]       # Delta_i for each object
    total_retained: float             # Expected total importance retained
    theoretical_optimal: float        # Theoretical maximum (with K=infinity)
    # Chain-level stats (optional)
    chain_count: int = 0              # Number of chains extracted
    chain_coverage: float = 0.0       # Ratio of chains selected
    orphan_count: int = 0             # Number of orphan objects added


class VAGE:
    """
    Vulnerability-Aware Greedy Extraction Algorithm.

    Theorem (Optimality): VAGE returns the globally optimal solution
    for the information retention maximization problem.

    Algorithm:
        1. For each candidate object, compute:
           Delta_i = omega(s_i) * (1 - rho(s_i))
                   = importance * vulnerability

        2. Sort by Delta_i descending

        3. Select top-K objects

    Time Complexity: O(N log N) for sorting
    """

    def __init__(self, config: Optional[VAGEConfig] = None):
        self.config = config or VAGEConfig()
        self.retention_predictor = RetentionPredictor(config)
        self.importance_estimator = ImportanceEstimator(
            source=self.config.importance_source
        )

    def prioritize(
        self,
        objects: List[any],  # List of CanvasObject
        total_turns: int,
        budget_k: Optional[int] = None,
    ) -> VAGEResult:
        """
        Prioritize objects for extraction using VAGE algorithm.

        Args:
            objects: List of candidate CanvasObjects
            total_turns: Total conversation turns so far
            budget_k: Maximum objects to select (None = select all)

        Returns:
            VAGEResult with selected indices and scores
        """
        if not objects:
            return VAGEResult(
                selected_indices=[],
                marginal_gains=[],
                total_retained=0.0,
                theoretical_optimal=0.0,
            )

        budget_k = budget_k or self.config.default_budget_k

        # Step 1: Compute marginal gains for each object
        gains = []
        for i, obj in enumerate(objects):
            # Get importance (omega)
            omega = self.importance_estimator.estimate(
                content=obj.content,
                confidence=getattr(obj, 'confidence', 1.0),
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
            )

            # Get retention probability (rho)
            rho = self.retention_predictor.predict(
                turn_id=getattr(obj, 'turn_id', total_turns),
                total_turns=total_turns,
                content=obj.content,
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
            )

            # Marginal gain: Delta = omega * (1 - rho)
            vulnerability = 1.0 - rho
            delta = omega * vulnerability

            gains.append((i, delta, omega, rho))

        # Save marginal gains in ORIGINAL order (before sorting)
        marginal_gains = [g[1] for g in gains]

        # Step 2: Sort by marginal gain (descending)
        gains.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Select top-K
        selected = gains[:budget_k]
        selected_indices = [g[0] for g in selected]

        # Compute metrics
        total_retained = sum(g[1] for g in selected)  # Sum of selected gains
        theoretical_optimal = sum(g[1] for g in gains)  # Sum of all gains (K=infinity)

        return VAGEResult(
            selected_indices=selected_indices,
            marginal_gains=marginal_gains,
            total_retained=total_retained,
            theoretical_optimal=theoretical_optimal,
        )

    def compute_object_scores(
        self,
        objects: List[any],
        total_turns: int,
    ) -> List[Tuple[float, float, float]]:
        """
        Compute (omega, rho, delta) for each object without selection.

        Useful for analysis and visualization.

        Returns:
            List of (importance, retention_prob, marginal_gain) tuples
        """
        scores = []
        for obj in objects:
            omega = self.importance_estimator.estimate(
                content=obj.content,
                confidence=getattr(obj, 'confidence', 1.0),
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
            )

            rho = self.retention_predictor.predict(
                turn_id=getattr(obj, 'turn_id', total_turns),
                total_turns=total_turns,
                content=obj.content,
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
            )

            delta = omega * (1.0 - rho)
            scores.append((omega, rho, delta))

        return scores

    def compute_object_scores_dict(
        self,
        objects: List[Any],
        total_turns: int,
    ) -> Dict[str, ObjectScore]:
        """
        Compute ObjectScore for each object, indexed by object ID.

        Returns:
            Dict mapping object ID to ObjectScore
        """
        scores = {}
        for obj in objects:
            omega = self.importance_estimator.estimate(
                content=obj.content,
                confidence=getattr(obj, 'confidence', 1.0),
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
            )

            rho = self.retention_predictor.predict(
                turn_id=getattr(obj, 'turn_id', total_turns),
                total_turns=total_turns,
                content=obj.content,
                obj_type=obj.type.value if hasattr(obj.type, 'value') else str(obj.type),
            )

            delta = omega * (1.0 - rho)
            scores[obj.id] = ObjectScore(omega=omega, rho=rho, delta=delta)

        return scores

    def _extract_causal_chains(
        self,
        objects: List[Any],
        graph: 'CanvasGraph',
        max_depth: int = 3,
        verbose: bool = False,
        all_objects: Optional[List[Any]] = None,  # All objects for traversal
    ) -> List[List[Any]]:
        """
        Extract causal chains by traversing caused_by edges backward from Decision nodes.

        Args:
            objects: List of candidate objects (only Decisions from here are used as starting points)
            graph: The canvas graph with edge information
            max_depth: Maximum chain depth to traverse
            all_objects: All objects available for chain traversal (if None, uses objects)

        Returns:
            List of chains, where each chain is a list of objects
            (starting with Decision, followed by its causes)
        """
        chains = []
        # Use all_objects for traversal if provided, otherwise just objects
        traversal_pool = all_objects if all_objects is not None else objects
        obj_map = {obj.id: obj for obj in traversal_pool}

        # Find all Decision objects (only from candidate objects, not traversal pool)
        decisions = []
        for obj in objects:
            obj_type = obj.type.value if hasattr(obj.type, 'value') else str(obj.type)
            if obj_type.upper() == 'DECISION':
                decisions.append(obj)

        if verbose:
            print(f"  Candidates: {len(objects)}, Traversal pool: {len(traversal_pool)}, Decisions: {len(decisions)}")

        # For each Decision, traverse backward via caused_by
        for decision in decisions:
            chain = [decision]
            visited = {decision.id}
            queue = [(decision.id, 0)]

            while queue:
                current_id, depth = queue.pop(0)
                if depth >= max_depth:
                    continue

                # Get caused_by neighbors (incoming edges)
                neighbors = graph.get_neighbors(
                    current_id,
                    relation='caused_by',
                    direction='outgoing'  # caused_by edges point FROM effect TO cause
                )

                for neighbor_id in neighbors:
                    if neighbor_id not in visited and neighbor_id in obj_map:
                        visited.add(neighbor_id)
                        chain.append(obj_map[neighbor_id])
                        queue.append((neighbor_id, depth + 1))

            # Only keep chains with at least 2 nodes
            if len(chain) >= 2:
                chains.append(chain)

        return chains

    def prioritize_chains(
        self,
        objects: List[Any],
        total_turns: int,
        budget_k: int,
        graph: 'CanvasGraph',
        max_chain_depth: int = 3,
        completeness_bonus: float = 0.2,
        verbose: bool = False,
        all_objects: Optional[List[Any]] = None,  # All objects for chain traversal
    ) -> VAGEResult:
        """
        Chain-Level VAGE: Select optimal chain combinations.

        Instead of selecting individual objects, this method:
        1. Extracts all causal chains (Decision -> Constraint -> ...)
        2. Scores each chain by total Δ × completeness bonus
        3. Greedily selects non-overlapping chains by efficiency
        4. Fills remaining budget with high-Δ orphan objects

        Args:
            objects: List of candidate objects
            total_turns: Total conversation turns
            budget_k: Maximum objects to select
            graph: Canvas graph with edge information
            max_chain_depth: Maximum chain traversal depth
            completeness_bonus: Bonus per chain node (default 0.2)
            verbose: Print detailed progress logs

        Returns:
            VAGEResult with chain-aware selection
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Chain-VAGE] Starting prioritize_chains")
            print(f"  Objects: {len(objects)}, Budget: {budget_k}, Turns: {total_turns}")
            print(f"{'='*60}")

        if not objects:
            if verbose:
                print("[Chain-VAGE] No objects to process, returning empty result")
            return VAGEResult(
                selected_indices=[],
                marginal_gains=[],
                total_retained=0.0,
                theoretical_optimal=0.0,
            )

        # Step 1: Compute scores for all objects
        if verbose:
            print(f"\n[Step 1] Computing scores for {len(objects)} objects...")
        obj_scores = self.compute_object_scores_dict(objects, total_turns)
        obj_to_idx = {obj.id: i for i, obj in enumerate(objects)}

        if verbose:
            # Show top 5 objects by delta
            sorted_objs = sorted(obj_scores.items(), key=lambda x: x[1].delta, reverse=True)[:5]
            print(f"  Top 5 objects by Δ:")
            for obj_id, score in sorted_objs:
                obj = next((o for o in objects if o.id == obj_id), None)
                obj_type = obj.type.value if obj and hasattr(obj.type, 'value') else 'unknown'
                content_preview = obj.content[:40] if obj else ''
                print(f"    [{obj_type}] Δ={score.delta:.3f} (ω={score.omega:.2f}, ρ={score.rho:.2f}): {content_preview}...")

        # Step 2: Extract causal chains
        if verbose:
            print(f"\n[Step 2] Extracting causal chains (max_depth={max_chain_depth})...")
        # Use all_objects for traversal if provided, otherwise just objects
        traversal_objects = all_objects if all_objects is not None else objects
        chains = self._extract_causal_chains(
            objects, graph, max_chain_depth, verbose=verbose, all_objects=traversal_objects
        )

        if verbose:
            print(f"  Found {len(chains)} causal chains")
            for i, chain in enumerate(chains[:3]):  # Show first 3 chains
                chain_types = [o.type.value if hasattr(o.type, 'value') else str(o.type) for o in chain]
                print(f"    Chain {i+1}: {' -> '.join(chain_types)} (len={len(chain)})")
            if len(chains) > 3:
                print(f"    ... and {len(chains)-3} more chains")

        # Step 3: Score each chain
        # Note: obj_scores only contains new objects, old objects in chain are excluded from score
        if verbose:
            print(f"\n[Step 3] Scoring chains...")
        chain_scores = []
        for chain in chains:
            # Only sum scores for objects we have scores for (new objects)
            base_score = sum(obj_scores[obj.id].delta for obj in chain if obj.id in obj_scores)
            bonus = 1.0 + completeness_bonus * len(chain)
            total_score = base_score * bonus
            efficiency = total_score / len(chain) if len(chain) > 0 else 0  # Score per object
            chain_scores.append((chain, efficiency, total_score))

        if verbose and chain_scores:
            # Sort and show top chains
            sorted_chains = sorted(chain_scores, key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3 chains by efficiency:")
            for chain, eff, score in sorted_chains:
                chain_types = [o.type.value if hasattr(o.type, 'value') else str(o.type) for o in chain]
                print(f"    {' -> '.join(chain_types)}: eff={eff:.3f}, score={score:.3f}")

        # Step 4: Greedy selection by efficiency (descending)
        if verbose:
            print(f"\n[Step 4] Greedy chain selection (budget={budget_k})...")
        chain_scores.sort(key=lambda x: x[1], reverse=True)

        selected = []
        selected_ids = set()
        total_score = 0.0
        chains_selected = 0

        # Build candidate IDs set (only new objects can be selected)
        candidate_ids = {obj.id for obj in objects}

        for chain, efficiency, score in chain_scores:
            chain_ids = {obj.id for obj in chain}
            # Only select objects that are in candidates (new objects)
            new_chain_objs = [obj for obj in chain if obj.id in candidate_ids]
            new_chain_ids = {obj.id for obj in new_chain_objs}

            # Skip if no new objects in chain
            if not new_chain_objs:
                continue

            # Skip if overlaps with already selected
            if new_chain_ids & selected_ids:
                if verbose:
                    print(f"    Skipped chain (overlap): len={len(new_chain_objs)} new objs")
                continue

            # Check budget (only count new objects)
            if len(selected) + len(new_chain_objs) <= budget_k:
                selected.extend(new_chain_objs)
                selected_ids.update(new_chain_ids)
                total_score += score
                chains_selected += 1
                if verbose:
                    chain_types = [o.type.value if hasattr(o.type, 'value') else str(o.type) for o in chain]
                    print(f"    Selected chain {chains_selected}: {' -> '.join(chain_types)} ({len(new_chain_objs)}/{len(chain)} new, score={score:.3f})")

        if verbose:
            print(f"  Selected {chains_selected} chains, {len(selected)} objects, score={total_score:.3f}")

        # Step 5: Fill remaining budget with high-Δ orphan objects
        remaining = budget_k - len(selected)
        orphan_count = 0

        if remaining > 0:
            if verbose:
                print(f"\n[Step 5] Filling {remaining} remaining slots with orphan objects...")
            orphans = [
                (obj, obj_scores[obj.id].delta)
                for obj in objects
                if obj.id not in selected_ids
            ]
            orphans.sort(key=lambda x: x[1], reverse=True)

            for obj, delta in orphans[:remaining]:
                selected.append(obj)
                total_score += delta
                orphan_count += 1
                if verbose:
                    obj_type = obj.type.value if hasattr(obj.type, 'value') else str(obj.type)
                    print(f"    Added orphan [{obj_type}]: Δ={delta:.3f}")

        # Build result
        marginal_gains = [obj_scores[obj.id].delta for obj in objects]
        theoretical_optimal = sum(s.delta for s in obj_scores.values())

        # Get indices in original order
        selected_indices = [obj_to_idx[obj.id] for obj in selected]

        if verbose:
            print(f"\n{'='*60}")
            print(f"[Chain-VAGE] Summary:")
            print(f"  Chains found: {len(chains)}")
            print(f"  Chains selected: {chains_selected}")
            print(f"  Orphans added: {orphan_count}")
            print(f"  Total selected: {len(selected_indices)}/{len(objects)}")
            print(f"  Score retained: {total_score:.3f}/{theoretical_optimal:.3f} ({100*total_score/theoretical_optimal if theoretical_optimal > 0 else 0:.1f}%)")
            print(f"{'='*60}\n")

        return VAGEResult(
            selected_indices=selected_indices,
            marginal_gains=marginal_gains,
            total_retained=total_score,
            theoretical_optimal=theoretical_optimal,
            chain_count=len(chains),
            chain_coverage=chains_selected / len(chains) if chains else 0.0,
            orphan_count=orphan_count,
        )


# Convenience function for quick usage
def vage_select(
    objects: List[any],
    total_turns: int,
    budget_k: int = 10,
    config: Optional[VAGEConfig] = None,
) -> List[int]:
    """
    Quick function to get selected indices using VAGE.

    Args:
        objects: List of CanvasObjects
        total_turns: Total turns in conversation
        budget_k: How many to select
        config: Optional VAGE configuration

    Returns:
        List of indices of selected objects (sorted by priority)
    """
    vage = VAGE(config)
    result = vage.prioritize(objects, total_turns, budget_k)
    return result.selected_indices


# For theoretical analysis
def compute_retention_bound(
    objects: List[any],
    total_turns: int,
    budget_k: int,
    config: Optional[VAGEConfig] = None,
) -> Tuple[float, float, float]:
    """
    Compute theoretical bounds from Theorem 2.

    Returns:
        (natural_retention, extraction_gain, total_retention)

    Where:
        natural_retention = sum(omega_i * rho_i)
        extraction_gain = sum(selected Delta_i)
        total_retention = natural + gain
    """
    vage = VAGE(config)
    scores = vage.compute_object_scores(objects, total_turns)

    # Natural retention (what survives without extraction)
    natural = sum(omega * rho for omega, rho, _ in scores)

    # Get selected objects
    result = vage.prioritize(objects, total_turns, budget_k)

    # Extraction gain
    gain = result.total_retained

    return natural, gain, natural + gain
