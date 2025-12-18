"""
LoCoMo Dataset Adapter for CogCanvas Evaluation.

Adapts the LoCoMo (Long Context Multi-hop) dataset to the CogCanvas evaluation format.

LoCoMo Dataset Format:
- Multi-session conversations between two speakers
- QA pairs with evidence references (dialogue IDs)
- Categories: 1=single-hop, 2=temporal, 3=multi-hop

Adaptation Strategy:
1. Flatten multi-session conversations into sequential turns
2. Map QA pairs to test questions with evidence tracking
3. Place compression point at middle of conversation
4. Support category-based filtering and analysis

Reference: LoCoMo is a benchmark for long-context multi-hop question answering
"""

import json
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from experiments.data_gen import ConversationTurn


# =============================================================================
# LoCoMo Data Structures
# =============================================================================

@dataclass
class LoCoMoQAPair:
    """A question-answer pair from LoCoMo dataset."""
    question: str
    answer: str
    evidence: List[str]  # List of dialogue IDs (e.g., ["D1:3", "D2:5"])
    category: int  # 1=single-hop, 2=temporal, 3=multi-hop

    @property
    def category_name(self) -> str:
        """Get human-readable category name."""
        return {1: "single-hop", 2: "temporal", 3: "multi-hop"}.get(self.category, "unknown")


@dataclass
class LoCoMoConversation:
    """A conversation from LoCoMo dataset in CogCanvas format."""
    id: str
    speaker_a: str
    speaker_b: str
    turns: List[ConversationTurn]
    qa_pairs: List[LoCoMoQAPair]
    dialogue_id_to_turn: Dict[str, int]  # Map dialogue ID to turn number
    metadata: Dict

    def get_compression_point(self) -> int:
        """Get suggested compression point (middle of conversation)."""
        return len(self.turns) // 2

    def get_qa_by_category(self, category: int) -> List[LoCoMoQAPair]:
        """Filter QA pairs by category."""
        return [qa for qa in self.qa_pairs if qa.category == category]


# =============================================================================
# Data Loading
# =============================================================================

def load_locomo(path: str) -> List[dict]:
    """
    Load LoCoMo dataset from JSON file.

    Args:
        path: Path to locomo.json file

    Returns:
        List of raw LoCoMo conversation dictionaries
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations from {path}")
    return data


# =============================================================================
# Format Conversion
# =============================================================================

def convert_to_eval_format(locomo_data: List[dict]) -> List[LoCoMoConversation]:
    """
    Convert LoCoMo data to CogCanvas evaluation format.

    Conversion steps:
    1. Flatten multi-session dialogues into sequential turns
    2. Alternate speakers as user/assistant
    3. Map dialogue IDs to turn numbers
    4. Convert QA pairs with evidence tracking

    Args:
        locomo_data: Raw LoCoMo data

    Returns:
        List of LoCoMoConversation objects
    """
    conversations = []

    for idx, raw_conv in enumerate(locomo_data):
        conv_data = raw_conv['conversation']
        qa_data = raw_conv['qa']

        # Extract speaker names
        speaker_a = conv_data['speaker_a']
        speaker_b = conv_data['speaker_b']

        # Flatten all sessions into sequential turns
        turns = []
        dialogue_id_to_turn = {}
        turn_id = 1

        # Extract all session keys and sort them
        session_keys = sorted([
            k for k in conv_data.keys()
            if k.startswith('session_') and not k.endswith('_date_time')
        ], key=lambda x: int(x.split('_')[1]))

        for session_key in session_keys:
            session_dialogues = conv_data[session_key]

            # Extract session datetime for this session
            session_num = session_key.split('_')[1]
            datetime_key = f"session_{session_num}_date_time"
            session_datetime = conv_data.get(datetime_key, None)

            for dialogue in session_dialogues:
                speaker = dialogue['speaker']
                text = dialogue['text']
                dia_id = dialogue['dia_id']

                # Determine if this speaker is "user" or "assistant"
                # We'll treat speaker_a as user, speaker_b as assistant
                # But alternate based on actual speaker in dialogue
                if speaker == speaker_a:
                    user_text = text
                    assistant_text = ""
                    # Mark as user turn
                    is_user_turn = True
                else:
                    user_text = ""
                    assistant_text = text
                    is_user_turn = False

                # For LoCoMo, we'll treat each dialogue as a turn
                # Speaker A's utterances are "user", Speaker B's are "assistant"
                if is_user_turn:
                    # If previous turn was also user, or this is first turn, create new turn
                    if not turns or turns[-1].assistant != "":
                        turns.append(ConversationTurn(
                            turn_id=turn_id,
                            user=user_text,
                            assistant="",
                            session_datetime=session_datetime
                        ))
                        dialogue_id_to_turn[dia_id] = turn_id
                        turn_id += 1
                    else:
                        # Append to previous user message
                        turns[-1].user += "\n" + user_text
                        dialogue_id_to_turn[dia_id] = turns[-1].turn_id
                else:
                    # Assistant response
                    if turns and turns[-1].assistant == "":
                        # Complete the current turn
                        turns[-1].assistant = assistant_text
                        dialogue_id_to_turn[dia_id] = turns[-1].turn_id
                    else:
                        # Create a new turn with empty user message
                        turns.append(ConversationTurn(
                            turn_id=turn_id,
                            user="",
                            assistant=assistant_text,
                            session_datetime=session_datetime
                        ))
                        dialogue_id_to_turn[dia_id] = turn_id
                        turn_id += 1

        # Handle case where conversation ends with user message
        for turn in turns:
            if turn.assistant == "":
                turn.assistant = "[No response]"
            if turn.user == "":
                turn.user = "[Continued]"

        # Convert QA pairs (skip those without answers)
        qa_pairs = [
            LoCoMoQAPair(
                question=qa['question'],
                answer=qa['answer'] if isinstance(qa['answer'], str) else str(qa['answer']),
                evidence=qa['evidence'],
                category=qa['category']
            )
            for qa in qa_data
            if 'answer' in qa and qa['answer']  # Only include QA pairs with answers
        ]

        # Create conversation ID
        conv_id = f"locomo_{idx:03d}"

        # Build metadata
        metadata = {
            'speaker_a': speaker_a,
            'speaker_b': speaker_b,
            'num_sessions': len(session_keys),
            'num_turns': len(turns),
            'num_qa_pairs': len(qa_pairs),
            'category_distribution': {
                'single-hop': len([qa for qa in qa_pairs if qa.category == 1]),
                'temporal': len([qa for qa in qa_pairs if qa.category == 2]),
                'multi-hop': len([qa for qa in qa_pairs if qa.category == 3]),
            }
        }

        conversations.append(LoCoMoConversation(
            id=conv_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            turns=turns,
            qa_pairs=qa_pairs,
            dialogue_id_to_turn=dialogue_id_to_turn,
            metadata=metadata
        ))

    print(f"Converted {len(conversations)} conversations")
    print(f"Total QA pairs: {sum(len(c.qa_pairs) for c in conversations)}")
    print(f"Average turns per conversation: {sum(len(c.turns) for c in conversations) / len(conversations):.1f}")

    return conversations


def verify_evidence_mapping(conversation: LoCoMoConversation) -> Dict[str, any]:
    """
    Verify that evidence dialogue IDs map correctly to turns.

    Args:
        conversation: LoCoMo conversation to verify

    Returns:
        Dictionary with verification statistics
    """
    total_evidence_refs = 0
    mapped_refs = 0
    unmapped_refs = []

    for qa in conversation.qa_pairs:
        for evidence_id in qa.evidence:
            total_evidence_refs += 1
            if evidence_id in conversation.dialogue_id_to_turn:
                mapped_refs += 1
            else:
                unmapped_refs.append((qa.question[:50], evidence_id))

    return {
        'conversation_id': conversation.id,
        'total_evidence_refs': total_evidence_refs,
        'mapped_refs': mapped_refs,
        'unmapped_refs': unmapped_refs,
        'mapping_rate': mapped_refs / total_evidence_refs if total_evidence_refs > 0 else 0.0
    }


# =============================================================================
# Export Functions
# =============================================================================

def export_to_json(conversations: List[LoCoMoConversation], output_path: str) -> None:
    """
    Export converted conversations to JSON format.

    Args:
        conversations: List of converted conversations
        output_path: Path to output JSON file
    """
    export_data = {
        'conversations': [
            {
                'id': conv.id,
                'speaker_a': conv.speaker_a,
                'speaker_b': conv.speaker_b,
                'turns': [
                    {
                        'turn_id': t.turn_id,
                        'user': t.user,
                        'assistant': t.assistant,
                        'session_datetime': t.session_datetime,
                    }
                    for t in conv.turns
                ],
                'qa_pairs': [
                    {
                        'question': qa.question,
                        'answer': qa.answer,
                        'evidence': qa.evidence,
                        'category': qa.category,
                        'category_name': qa.category_name,
                    }
                    for qa in conv.qa_pairs
                ],
                'dialogue_id_to_turn': conv.dialogue_id_to_turn,
                'metadata': conv.metadata,
            }
            for conv in conversations
        ],
        'dataset_info': {
            'name': 'LoCoMo',
            'num_conversations': len(conversations),
            'total_turns': sum(len(c.turns) for c in conversations),
            'total_qa_pairs': sum(len(c.qa_pairs) for c in conversations),
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(conversations)} conversations to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for converting LoCoMo dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert LoCoMo dataset to CogCanvas format")
    parser.add_argument(
        "--input", "-i",
        default="experiments/data/locomo10.json",
        help="Path to LoCoMo JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments/data/locomo_converted.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify evidence mapping"
    )

    args = parser.parse_args()

    # Load and convert
    print(f"Loading LoCoMo dataset from {args.input}...")
    raw_data = load_locomo(args.input)

    print("\nConverting to evaluation format...")
    conversations = convert_to_eval_format(raw_data)

    # Verify evidence mapping if requested
    if args.verify:
        print("\nVerifying evidence mapping...")
        for conv in conversations:
            stats = verify_evidence_mapping(conv)
            print(f"  {stats['conversation_id']}: {stats['mapped_refs']}/{stats['total_evidence_refs']} "
                  f"({stats['mapping_rate']:.1%}) evidence refs mapped")
            if stats['unmapped_refs']:
                print(f"    Unmapped: {stats['unmapped_refs'][:3]}")  # Show first 3

    # Export
    print(f"\nExporting to {args.output}...")
    export_to_json(conversations, args.output)

    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total conversations: {len(conversations)}")
    print(f"Total turns: {sum(len(c.turns) for c in conversations)}")
    print(f"Total QA pairs: {sum(len(c.qa_pairs) for c in conversations)}")
    print(f"Average turns/conversation: {sum(len(c.turns) for c in conversations) / len(conversations):.1f}")
    print(f"Average QA pairs/conversation: {sum(len(c.qa_pairs) for c in conversations) / len(conversations):.1f}")

    # Category distribution
    from collections import defaultdict
    total_by_category = defaultdict(int)
    for conv in conversations:
        for qa in conv.qa_pairs:
            total_by_category[qa.category] += 1

    print("\nQA Category Distribution:")
    for cat in sorted(total_by_category.keys()):
        cat_name = {
            1: "Single-hop",
            2: "Temporal",
            3: "Multi-hop",
            4: "Category-4",
            5: "Category-5"
        }.get(cat, f"Category-{cat}")
        print(f"  {cat_name}: {total_by_category[cat]}")


if __name__ == "__main__":
    main()
