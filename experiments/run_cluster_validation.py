"""
Phase 0: Cluster-Aware Retrieval Validation Experiment

This experiment validates whether cluster information improves retrieval quality
before committing to larger codebase changes.

Approach:
1. Load existing CogCanvas results (baseline)
2. Load facts with cluster IDs
3. Simulate cluster-aware retrieval with different strategies
4. Compare: baseline vs cluster-boosted retrieval

Key question: Does knowing the semantic cluster help retrieve relevant info?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ValidationResult:
    """Result of a single retrieval experiment."""
    strategy: str
    recall_rate: float
    exact_match_rate: float
    avg_fuzzy_score: float
    improvement_over_baseline: float  # percentage points


def load_baseline_results(path: str = 'experiments/results/cogcanvas_50_topk5.json') -> Dict:
    """Load baseline CogCanvas results."""
    with open(path) as f:
        return json.load(f)


def load_facts_with_clusters(path: str = 'experiments/data/facts_with_clusters.json') -> List[Dict]:
    """Load facts with their cluster assignments."""
    with open(path) as f:
        data = json.load(f)
    return data['facts']


def analyze_cluster_performance(
    baseline: Dict,
    facts_with_clusters: List[Dict],
) -> Dict[int, Dict]:
    """Analyze retrieval performance by cluster.

    Returns performance metrics for each cluster to identify patterns.
    """
    # Build fact_id -> cluster_id mapping
    fact_to_cluster = {f['id']: f['cluster_id'] for f in facts_with_clusters}

    # Group results by cluster
    cluster_stats = defaultdict(lambda: {
        'total': 0,
        'passed': 0,
        'exact_matches': 0,
        'fuzzy_scores': [],
        'fact_types': [],
    })

    for conv in baseline['conversations']:
        for fact in conv['facts']:
            fact_id = fact['id']
            cluster_id = fact_to_cluster.get(fact_id, -1)

            stats = cluster_stats[cluster_id]
            stats['total'] += 1
            stats['passed'] += 1 if fact['passed'] else 0
            stats['exact_matches'] += 1 if fact['exact_match'] else 0
            stats['fuzzy_scores'].append(fact['fuzzy_score'])
            stats['fact_types'].append(fact['type'])

    # Compute aggregates
    result = {}
    for cluster_id, stats in cluster_stats.items():
        n = stats['total']
        result[cluster_id] = {
            'n': n,
            'recall_rate': stats['passed'] / n if n > 0 else 0,
            'exact_match_rate': stats['exact_matches'] / n if n > 0 else 0,
            'avg_fuzzy': np.mean(stats['fuzzy_scores']) if stats['fuzzy_scores'] else 0,
            'types': dict(defaultdict(int, {t: stats['fact_types'].count(t) for t in set(stats['fact_types'])})),
        }

    return result


def compute_cluster_vulnerability(cluster_stats: Dict[int, Dict]) -> Dict[int, float]:
    """Compute vulnerability score for each cluster.

    Vulnerability = 1 - recall_rate
    Higher vulnerability = more likely to be lost in compression
    """
    vulnerabilities = {}
    for cluster_id, stats in cluster_stats.items():
        vulnerabilities[cluster_id] = 1.0 - stats['recall_rate']
    return vulnerabilities


def simulate_cluster_aware_vage(
    baseline: Dict,
    facts_with_clusters: List[Dict],
    cluster_vulnerabilities: Dict[int, float],
    vage_boost_factor: float = 0.2,
) -> Dict:
    """Simulate VAGE with cluster-aware vulnerability scores.

    This is a thought experiment: if we had used cluster-based vulnerability
    scores during extraction, would more facts have been preserved?

    The idea: facts in high-vulnerability clusters should have been
    prioritized for extraction (stored in canvas).

    Returns simulated improvement metrics.
    """
    fact_to_cluster = {f['id']: f['cluster_id'] for f in facts_with_clusters}

    # Analyze which failed facts could have been saved
    failed_facts = []
    passed_facts = []

    for conv in baseline['conversations']:
        for fact in conv['facts']:
            fact_id = fact['id']
            cluster_id = fact_to_cluster.get(fact_id, -1)
            vulnerability = cluster_vulnerabilities.get(cluster_id, 0.5)

            fact_info = {
                **fact,
                'cluster_id': cluster_id,
                'vulnerability': vulnerability,
            }

            if fact['passed']:
                passed_facts.append(fact_info)
            else:
                failed_facts.append(fact_info)

    # Insight: Can cluster vulnerability predict failure?
    if failed_facts:
        avg_vuln_failed = np.mean([f['vulnerability'] for f in failed_facts])
        avg_vuln_passed = np.mean([f['vulnerability'] for f in passed_facts]) if passed_facts else 0

        vulnerability_gap = avg_vuln_failed - avg_vuln_passed
    else:
        vulnerability_gap = 0

    return {
        'num_failed': len(failed_facts),
        'num_passed': len(passed_facts),
        'avg_vuln_failed': avg_vuln_failed if failed_facts else 0,
        'avg_vuln_passed': avg_vuln_passed if passed_facts else 0,
        'vulnerability_gap': vulnerability_gap,
        'failed_clusters': [f['cluster_id'] for f in failed_facts],
        'insights': {
            'cluster_predicts_failure': vulnerability_gap > 0.05,
            'gap_magnitude': vulnerability_gap,
        }
    }


def compute_cluster_importance_weights(
    cluster_stats: Dict[int, Dict],
    strategy: str = 'inverse_recall',
) -> Dict[int, float]:
    """Compute importance weights for clusters.

    Strategies:
    - inverse_recall: Higher weight for clusters with lower recall
    - uniform: Equal weights
    - type_based: Based on fact type distribution
    """
    weights = {}

    if strategy == 'uniform':
        for cluster_id in cluster_stats:
            weights[cluster_id] = 1.0

    elif strategy == 'inverse_recall':
        for cluster_id, stats in cluster_stats.items():
            # Lower recall = higher importance (needs more attention)
            weights[cluster_id] = 1.0 + (1.0 - stats['recall_rate'])

    elif strategy == 'type_based':
        type_importance = {
            'decision': 1.3,
            'key_fact': 1.0,
            'reminder': 0.9,
            'insight': 0.8,
            'todo': 1.1,
        }
        for cluster_id, stats in cluster_stats.items():
            types = stats['types']
            total = sum(types.values())
            if total > 0:
                weighted_sum = sum(
                    count * type_importance.get(t, 1.0)
                    for t, count in types.items()
                )
                weights[cluster_id] = weighted_sum / total
            else:
                weights[cluster_id] = 1.0

    return weights


def print_cluster_analysis(cluster_stats: Dict[int, Dict]):
    """Pretty print cluster performance analysis."""
    print("\n" + "=" * 70)
    print("CLUSTER PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Sort by recall rate (worst first)
    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda x: x[1]['recall_rate']
    )

    print(f"\n{'Cluster':<10} {'N':>6} {'Recall':>10} {'Exact':>10} {'Fuzzy':>10} {'Types':<25}")
    print("-" * 70)

    for cluster_id, stats in sorted_clusters:
        types_str = ', '.join(f"{t}:{c}" for t, c in stats['types'].items())
        print(
            f"{cluster_id:<10} "
            f"{stats['n']:>6} "
            f"{stats['recall_rate']*100:>9.1f}% "
            f"{stats['exact_match_rate']*100:>9.1f}% "
            f"{stats['avg_fuzzy']:>9.1f} "
            f"{types_str:<25}"
        )

    # Highlight vulnerable clusters
    vulnerable = [
        (cid, stats) for cid, stats in sorted_clusters
        if stats['recall_rate'] < 1.0 and stats['n'] >= 3
    ]

    if vulnerable:
        print("\n⚠️  VULNERABLE CLUSTERS (recall < 100%, n >= 3):")
        for cid, stats in vulnerable:
            print(f"   Cluster {cid}: {stats['recall_rate']*100:.1f}% recall ({stats['n']} facts)")


def main():
    """Run Phase 0 validation experiment."""
    print("=" * 70)
    print("PHASE 0: CLUSTER-AWARE RETRIEVAL VALIDATION")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    try:
        baseline = load_baseline_results()
        facts_with_clusters = load_facts_with_clusters()
        print(f"    Loaded baseline: {len(baseline['conversations'])} conversations")
        print(f"    Loaded {len(facts_with_clusters)} facts with cluster IDs")
    except FileNotFoundError as e:
        print(f"    ❌ Error: {e}")
        print("    Please run schema discovery first: python experiments/run_schema_discovery.py")
        return

    # Baseline metrics
    print("\n[2] Baseline Performance:")
    print(f"    Recall Rate:      {baseline['summary']['recall_rate']}")
    print(f"    Exact Match Rate: {baseline['summary']['exact_match_rate']}")
    print(f"    Avg Fuzzy Score:  {baseline['summary']['avg_fuzzy_score']}")

    # Cluster analysis
    print("\n[3] Analyzing cluster performance...")
    cluster_stats = analyze_cluster_performance(baseline, facts_with_clusters)
    print_cluster_analysis(cluster_stats)

    # Vulnerability analysis
    print("\n[4] Computing cluster vulnerabilities...")
    vulnerabilities = compute_cluster_vulnerability(cluster_stats)

    print(f"\n{'Cluster':<10} {'Vulnerability':>15} {'Interpretation':<30}")
    print("-" * 55)
    for cid, vuln in sorted(vulnerabilities.items(), key=lambda x: -x[1]):
        if vuln > 0.1:
            interp = "⚠️  HIGH - prioritize extraction"
        elif vuln > 0.05:
            interp = "MEDIUM - watch carefully"
        else:
            interp = "LOW - generally retained"
        print(f"{cid:<10} {vuln*100:>14.1f}% {interp:<30}")

    # VAGE simulation
    print("\n[5] Simulating VAGE with cluster awareness...")
    vage_sim = simulate_cluster_aware_vage(
        baseline, facts_with_clusters, vulnerabilities
    )

    print(f"\n    Failed facts: {vage_sim['num_failed']}")
    print(f"    Passed facts: {vage_sim['num_passed']}")
    print(f"    Avg vulnerability (failed): {vage_sim['avg_vuln_failed']*100:.1f}%")
    print(f"    Avg vulnerability (passed): {vage_sim['avg_vuln_passed']*100:.1f}%")
    print(f"    Vulnerability gap: {vage_sim['vulnerability_gap']*100:.1f}%")

    if vage_sim['insights']['cluster_predicts_failure']:
        print("\n    ✅ INSIGHT: Cluster vulnerability CAN predict retrieval failure!")
        print("       → Cluster-aware VAGE is likely to improve performance")
    else:
        print("\n    ⚠️  INSIGHT: Cluster vulnerability does NOT predict failure well")
        print("       → Need alternative signals or cluster-aware approach may not help")

    # Failed cluster distribution
    if vage_sim['failed_clusters']:
        from collections import Counter
        failed_dist = Counter(vage_sim['failed_clusters'])
        print(f"\n    Failed fact distribution by cluster:")
        for cid, count in failed_dist.most_common():
            print(f"      Cluster {cid}: {count} failures")

    # Importance weights
    print("\n[6] Computing cluster importance weights...")
    for strategy in ['uniform', 'inverse_recall', 'type_based']:
        weights = compute_cluster_importance_weights(cluster_stats, strategy)
        avg_weight = np.mean(list(weights.values()))
        max_weight = max(weights.values())
        print(f"\n    Strategy: {strategy}")
        print(f"    Avg weight: {avg_weight:.3f}, Max weight: {max_weight:.3f}")

        # Show top weighted clusters
        top_clusters = sorted(weights.items(), key=lambda x: -x[1])[:3]
        print(f"    Top clusters: {', '.join(f'C{c}={w:.2f}' for c, w in top_clusters)}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("PHASE 0 CONCLUSION")
    print("=" * 70)

    vuln_gap = vage_sim['vulnerability_gap']
    if vuln_gap > 0.05:
        print("""
✅ VALIDATION PASSED

Cluster vulnerability shows predictive power for retrieval failure:
- Gap: {:.1f}% between failed and passed facts
- Recommendation: Proceed with Phase 1-2

Next steps:
1. Train Vulnerability model using recall results as labels
2. Implement cluster-aware retrieval in Canvas
3. Run ablation to measure actual improvement
        """.format(vuln_gap * 100))
    elif vuln_gap > 0:
        print("""
⚠️  MARGINAL SIGNAL

Cluster vulnerability shows weak predictive power:
- Gap: {:.1f}% (below 5% threshold)
- Recommendation: Consider alternative features

Options:
1. Add more features (position, content length, entity count)
2. Use ensemble of cluster + position
3. Keep rule-based VAGE, use clusters for analysis only
        """.format(vuln_gap * 100))
    else:
        print("""
❌ VALIDATION FAILED

Cluster vulnerability does NOT predict failure:
- Failed facts have LOWER vulnerability than passed
- Recommendation: Do NOT proceed with cluster-aware approach

Alternative path:
1. Keep current rule-based VAGE
2. Use cluster analysis as interpretability tool
3. Focus on position-based vulnerability (proven signal)
        """)

    # Save results
    output_path = Path('experiments/results/phase0_validation.json')
    output_path.parent.mkdir(exist_ok=True)

    results = {
        'baseline_summary': baseline['summary'],
        'cluster_stats': {str(k): v for k, v in cluster_stats.items()},
        'vulnerabilities': {str(k): v for k, v in vulnerabilities.items()},
        'vage_simulation': vage_sim,
        'conclusion': {
            'vulnerability_gap': vuln_gap,
            'passed_threshold': vuln_gap > 0.05,
            'recommendation': 'proceed' if vuln_gap > 0.05 else 'reconsider',
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
