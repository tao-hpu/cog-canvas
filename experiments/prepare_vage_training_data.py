"""
Prepare training data for VAGE (Vulnerability-Aware Greedy Extraction).

This script extracts training data from compression experiments:
- Features: position, length, has_numbers, has_entities, fact_type, etc.
- Label: was_lost (1 if info was lost after compression, 0 otherwise)

Compression Configuration:
- compression_turn = 40
- retain_recent = 5
- Info at turn < 35 will be lost after compression
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TrainingSample:
    """A single training sample for vulnerability prediction."""
    # Identifiers
    conversation_id: str
    fact_id: str

    # Features
    turn_id: int
    total_turns: int
    position_ratio: float  # turn_id / total_turns

    content_length: int
    quote_length: int

    has_numbers: bool
    has_named_entities: bool

    fact_type: str  # decision, key_fact, reminder, insight, todo
    difficulty: str  # easy, medium, hard, adversarial

    # For embedding-based features (optional)
    content: str
    quote: str

    # Label
    was_lost: bool  # True if this info was lost after compression


def has_numbers(text: str) -> bool:
    """Check if text contains numbers."""
    return bool(re.search(r'\d+', text))


def has_named_entities(text: str) -> bool:
    """Simple heuristic: capitalized words that aren't sentence starters."""
    words = text.split()
    for i, word in enumerate(words):
        # Skip first word of sentences
        if i > 0 and word and word[0].isupper() and not word.isupper():
            # Check it's not just after a period
            prev = words[i-1] if i > 0 else ""
            if not prev.endswith('.'):
                return True
    return False


def extract_samples_from_conversation(
    conv: Dict[str, Any],
    compression_turn: int = 40,
    retain_recent: int = 5,
    total_turns: int = 50,
) -> List[TrainingSample]:
    """Extract training samples from a single conversation."""
    samples = []

    # Threshold: info at turn < this will be lost
    loss_threshold = compression_turn - retain_recent  # 35

    for fact in conv.get('planted_facts', []):
        turn_id = fact['turn_id']

        # Determine if this fact was lost
        was_lost = turn_id < loss_threshold

        content = fact.get('content', '')
        quote = fact.get('quote', '')

        sample = TrainingSample(
            conversation_id=conv['id'],
            fact_id=fact['id'],
            turn_id=turn_id,
            total_turns=total_turns,
            position_ratio=turn_id / total_turns,
            content_length=len(content),
            quote_length=len(quote),
            has_numbers=has_numbers(content + quote),
            has_named_entities=has_named_entities(quote),
            fact_type=fact.get('fact_type', 'unknown'),
            difficulty=fact.get('difficulty', 'unknown'),
            content=content,
            quote=quote,
            was_lost=was_lost,
        )
        samples.append(sample)

    return samples


def load_and_extract(
    data_path: str,
    compression_turn: int = 40,
    retain_recent: int = 5,
    total_turns: int = 50,
) -> List[TrainingSample]:
    """Load dataset and extract all training samples."""
    with open(data_path) as f:
        data = json.load(f)

    all_samples = []
    for conv in data['conversations']:
        samples = extract_samples_from_conversation(
            conv, compression_turn, retain_recent, total_turns
        )
        all_samples.extend(samples)

    return all_samples


def samples_to_features(samples: List[TrainingSample]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert samples to feature matrix X and label vector y."""
    X = []
    y = []

    # One-hot encoding for fact_type
    fact_types = ['decision', 'key_fact', 'reminder', 'insight', 'todo', 'unknown']
    difficulties = ['easy', 'medium', 'hard', 'adversarial', 'unknown']

    for sample in samples:
        features = [
            sample.position_ratio,          # 0: position (0-1)
            sample.content_length / 100,    # 1: normalized content length
            sample.quote_length / 500,      # 2: normalized quote length
            float(sample.has_numbers),      # 3: has numbers
            float(sample.has_named_entities),  # 4: has entities
        ]

        # One-hot for fact_type
        for ft in fact_types:
            features.append(1.0 if sample.fact_type == ft else 0.0)

        # One-hot for difficulty
        for diff in difficulties:
            features.append(1.0 if sample.difficulty == diff else 0.0)

        X.append(features)
        y.append(1 if sample.was_lost else 0)

    return np.array(X), np.array(y)


def main():
    """Main function to prepare training data."""
    base_path = Path(__file__).parent

    # Load different datasets
    datasets = [
        ('eval_set.json', 40, 5, 50),           # Standard
        ('eval_set_hard.json', 40, 5, 50),      # Hard mode
        ('eval_set_long.json', 180, 5, 200),    # Long conversations
    ]

    all_samples = []

    for filename, comp_turn, retain, total in datasets:
        data_path = base_path / 'data' / filename
        if data_path.exists():
            samples = load_and_extract(
                str(data_path), comp_turn, retain, total
            )
            print(f"Loaded {len(samples)} samples from {filename}")
            all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples)}")

    # Analyze class distribution
    lost_count = sum(1 for s in all_samples if s.was_lost)
    retained_count = len(all_samples) - lost_count
    print(f"Lost: {lost_count} ({lost_count/len(all_samples)*100:.1f}%)")
    print(f"Retained: {retained_count} ({retained_count/len(all_samples)*100:.1f}%)")

    # Convert to features
    X, y = samples_to_features(all_samples)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")

    # Save training data
    output_path = base_path / 'data' / 'vage_training_data.json'
    output = {
        'samples': [asdict(s) for s in all_samples],
        'feature_names': [
            'position_ratio', 'content_length_norm', 'quote_length_norm',
            'has_numbers', 'has_named_entities',
            'type_decision', 'type_key_fact', 'type_reminder',
            'type_insight', 'type_todo', 'type_unknown',
            'diff_easy', 'diff_medium', 'diff_hard',
            'diff_adversarial', 'diff_unknown',
        ],
        'X': X.tolist(),
        'y': y.tolist(),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved training data to {output_path}")

    # Also save numpy arrays for quick loading
    np.save(base_path / 'data' / 'vage_X.npy', X)
    np.save(base_path / 'data' / 'vage_y.npy', y)
    print(f"Saved numpy arrays to vage_X.npy and vage_y.npy")

    return all_samples, X, y


if __name__ == '__main__':
    main()
