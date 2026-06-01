"""
LongMemEval Dataset Adapter for CogCanvas Evaluation.

Adapts the LongMemEval (ICLR 2025) benchmark dataset to the CogCanvas evaluation format.

LongMemEval Dataset Format:
- 500 questions, each with its own haystack of ~48 multi-session conversations
- QA pairs testing 5 memory ability categories across 6 question types
- Categories: 1=information-extraction, 2=multi-session-reasoning,
              3=knowledge-update, 4=temporal-reasoning, 5=abstention

Adaptation Strategy:
1. Each question + its haystack becomes one LongMemEvalConversation
2. Flatten haystack sessions into sequential ConversationTurn objects
3. Track session boundaries and evidence turns (has_answer=True)
4. Map question types to memory ability categories (1-5)
5. Support category-based filtering and evidence verification

Reference: LongMemEval is a benchmark for evaluating long-term memory
in chat assistants (ICLR 2025).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from experiments.data_gen import ConversationTurn


# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

QUESTION_TYPE_TO_CATEGORY: Dict[str, int] = {
    "single-session-user": 1,
    "single-session-assistant": 1,
    "single-session-preference": 1,
    "multi-session": 2,
    "knowledge-update": 3,
    "temporal-reasoning": 4,
}

CATEGORY_NAMES: Dict[int, str] = {
    1: "information-extraction",
    2: "multi-session-reasoning",
    3: "knowledge-update",
    4: "temporal-reasoning",
    5: "abstention",
}


# =============================================================================
# LongMemEval Data Structures
# =============================================================================

@dataclass
class LongMemEvalQAPair:
    """A question-answer pair from LongMemEval dataset."""
    question_id: str
    question: str
    answer: str  # Always stored as string (integers converted)
    question_type: str
    question_date: str
    answer_session_ids: List[str]
    is_abstention: bool

    @property
    def category(self) -> int:
        """Get numeric category (1-5) based on question type and abstention status."""
        if self.is_abstention:
            return 5
        return QUESTION_TYPE_TO_CATEGORY.get(self.question_type, 0)

    @property
    def category_name(self) -> str:
        """Get human-readable category name."""
        return CATEGORY_NAMES.get(self.category, "unknown")


@dataclass
class LongMemEvalConversation:
    """A conversation from LongMemEval dataset in CogCanvas format.

    Each LongMemEval question and its haystack sessions form one conversation.
    500 questions yield 500 conversations, each with exactly 1 QA pair.
    """
    id: str  # question_id
    turns: List[ConversationTurn]
    qa_pairs: List[LongMemEvalQAPair]  # Always exactly 1 element
    session_boundaries: List[int]  # turn_ids where new sessions start
    answer_turn_ids: List[int]  # turn_ids with has_answer=True
    metadata: Dict

    def get_compression_point(self) -> int:
        """Get suggested compression point (middle of conversation)."""
        return len(self.turns) // 2

    def get_qa_by_category(self, category: int) -> List[LongMemEvalQAPair]:
        """Filter QA pairs by category."""
        return [qa for qa in self.qa_pairs if qa.category == category]


# =============================================================================
# Data Loading
# =============================================================================

def load_longmemeval(path: str) -> List[dict]:
    """
    Load LongMemEval dataset from JSON file.

    Args:
        path: Path to longmemeval JSON file (e.g., longmemeval_s_cleaned.json)

    Returns:
        List of raw LongMemEval question dictionaries

    Raises:
        FileNotFoundError: If the input file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"LongMemEval data file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top level, got {type(data).__name__}")

    print(f"Loaded {len(data)} questions from {path}")
    return data


# =============================================================================
# Format Conversion
# =============================================================================

def convert_to_eval_format(
    longmemeval_data: List[dict],
    max_sessions: int = 0,
) -> List[LongMemEvalConversation]:
    """
    Convert LongMemEval data to CogCanvas evaluation format.

    Conversion steps:
    1. For each question, flatten its haystack sessions into sequential turns
    2. Pair user/assistant messages within each session
    3. Track session boundaries and evidence turns (has_answer=True)
    4. Create a single QA pair per conversation

    Args:
        longmemeval_data: Raw LongMemEval data (list of question dicts)
        max_sessions: Maximum number of sessions to include per conversation.
            0 means include all sessions (default).

    Returns:
        List of LongMemEvalConversation objects
    """
    conversations = []

    for idx, raw_item in enumerate(longmemeval_data):
        question_id = raw_item.get('question_id', f'unknown_{idx}')

        try:
            conv = _convert_single_item(raw_item, max_sessions)
            conversations.append(conv)
        except Exception as e:
            logger.warning(
                "Failed to convert question %s (index %d): %s",
                question_id, idx, e,
            )
            continue

    print(f"Converted {len(conversations)} conversations")
    print(f"Total QA pairs: {sum(len(c.qa_pairs) for c in conversations)}")
    if conversations:
        avg_turns = sum(len(c.turns) for c in conversations) / len(conversations)
        print(f"Average turns per conversation: {avg_turns:.1f}")

    return conversations


def _convert_single_item(
    raw_item: dict,
    max_sessions: int = 0,
) -> LongMemEvalConversation:
    """
    Convert a single LongMemEval question + haystack into a conversation.

    Args:
        raw_item: A single LongMemEval question dictionary
        max_sessions: Maximum sessions to include (0 = all)

    Returns:
        A LongMemEvalConversation object
    """
    question_id = raw_item['question_id']
    question_type = raw_item['question_type']
    question = raw_item['question']
    raw_answer = raw_item['answer']
    question_date = raw_item['question_date']
    haystack_dates = raw_item.get('haystack_dates', [])
    haystack_session_ids = raw_item.get('haystack_session_ids', [])
    haystack_sessions = raw_item.get('haystack_sessions', [])
    answer_session_ids = raw_item.get('answer_session_ids', [])

    # Determine abstention status from question_id suffix
    is_abstention = question_id.endswith('_abs')

    # Convert answer to string
    answer = str(raw_answer)

    # Apply max_sessions limit
    num_sessions = len(haystack_sessions)
    if max_sessions > 0 and num_sessions > max_sessions:
        haystack_sessions = haystack_sessions[:max_sessions]
        haystack_dates = haystack_dates[:max_sessions]
        haystack_session_ids = haystack_session_ids[:max_sessions]
        num_sessions = max_sessions

    # Flatten sessions into sequential turns
    turns: List[ConversationTurn] = []
    session_boundaries: List[int] = []
    answer_turn_ids: List[int] = []
    turn_id = 1

    # Build set of answer session IDs for fast lookup
    answer_session_id_set = set(answer_session_ids)

    for session_idx, session_messages in enumerate(haystack_sessions):
        # Extract session datetime
        session_datetime = (
            haystack_dates[session_idx]
            if session_idx < len(haystack_dates)
            else None
        )

        # Extract session ID for evidence tracking
        session_id = (
            haystack_session_ids[session_idx]
            if session_idx < len(haystack_session_ids)
            else None
        )
        is_evidence_session = session_id in answer_session_id_set

        # Record session boundary
        session_boundaries.append(turn_id)

        # Process messages within this session
        # Messages alternate user/assistant; pair them into ConversationTurn objects
        msg_idx = 0
        while msg_idx < len(session_messages):
            msg = session_messages[msg_idx]
            role = msg.get('role', '')
            content = msg.get('content', '')
            has_answer = msg.get('has_answer', False)

            if role == 'user':
                user_text = content
                assistant_text = ""
                user_has_answer = has_answer

                # Look ahead for the paired assistant response
                if msg_idx + 1 < len(session_messages):
                    next_msg = session_messages[msg_idx + 1]
                    if next_msg.get('role') == 'assistant':
                        assistant_text = next_msg.get('content', '')
                        assistant_has_answer = next_msg.get('has_answer', False)
                        msg_idx += 1  # Consume the assistant message
                    else:
                        assistant_has_answer = False
                else:
                    assistant_has_answer = False

                # Mark empty assistant response
                if not assistant_text:
                    assistant_text = "[No response]"

                turn = ConversationTurn(
                    turn_id=turn_id,
                    user=user_text,
                    assistant=assistant_text,
                    session_datetime=session_datetime,
                )
                turns.append(turn)

                # Track evidence turns
                if is_evidence_session and (user_has_answer or assistant_has_answer):
                    answer_turn_ids.append(turn_id)

                turn_id += 1

            elif role == 'assistant':
                # Assistant message without a preceding user message
                turn = ConversationTurn(
                    turn_id=turn_id,
                    user="[Continued]",
                    assistant=content,
                    session_datetime=session_datetime,
                )
                turns.append(turn)

                if is_evidence_session and has_answer:
                    answer_turn_ids.append(turn_id)

                turn_id += 1

            else:
                # Unknown role; skip with warning
                logger.warning(
                    "Unknown role '%s' in question %s, session %d, message %d",
                    role, question_id, session_idx, msg_idx,
                )

            msg_idx += 1

    # Create QA pair
    qa_pair = LongMemEvalQAPair(
        question_id=question_id,
        question=question,
        answer=answer,
        question_type=question_type,
        question_date=question_date,
        answer_session_ids=answer_session_ids,
        is_abstention=is_abstention,
    )

    # Build metadata
    metadata = {
        'question_type': question_type,
        'question_date': question_date,
        'is_abstention': is_abstention,
        'num_sessions': num_sessions,
        'num_turns': len(turns),
        'num_evidence_sessions': len(answer_session_ids),
        'num_evidence_turns': len(answer_turn_ids),
        'category': qa_pair.category,
        'category_name': qa_pair.category_name,
    }

    return LongMemEvalConversation(
        id=question_id,
        turns=turns,
        qa_pairs=[qa_pair],
        session_boundaries=session_boundaries,
        answer_turn_ids=answer_turn_ids,
        metadata=metadata,
    )


# =============================================================================
# Verification
# =============================================================================

def verify_evidence_mapping(conversation: LongMemEvalConversation) -> Dict[str, Any]:
    """
    Verify that evidence sessions map correctly to answer turns.

    Checks that conversations from answer_session_ids contain turns
    marked with has_answer=True, and reports coverage statistics.

    Args:
        conversation: LongMemEval conversation to verify

    Returns:
        Dictionary with verification statistics
    """
    qa = conversation.qa_pairs[0] if conversation.qa_pairs else None
    if qa is None:
        return {
            'conversation_id': conversation.id,
            'total_evidence_sessions': 0,
            'evidence_turns_found': 0,
            'session_boundaries': len(conversation.session_boundaries),
            'has_evidence_turns': False,
            'status': 'no_qa_pair',
        }

    total_evidence_sessions = len(qa.answer_session_ids)
    evidence_turns_found = len(conversation.answer_turn_ids)
    has_evidence_turns = evidence_turns_found > 0

    # For non-abstention questions, we expect at least one evidence turn
    if not qa.is_abstention and not has_evidence_turns:
        status = 'warning_no_evidence_turns'
    elif qa.is_abstention and has_evidence_turns:
        status = 'warning_abstention_has_evidence'
    else:
        status = 'ok'

    return {
        'conversation_id': conversation.id,
        'question_type': qa.question_type,
        'category': qa.category,
        'category_name': qa.category_name,
        'is_abstention': qa.is_abstention,
        'total_evidence_sessions': total_evidence_sessions,
        'evidence_turns_found': evidence_turns_found,
        'answer_turn_ids': conversation.answer_turn_ids,
        'total_turns': len(conversation.turns),
        'total_sessions': len(conversation.session_boundaries),
        'has_evidence_turns': has_evidence_turns,
        'status': status,
    }


# =============================================================================
# Export Functions
# =============================================================================

def export_to_json(conversations: List[LongMemEvalConversation], output_path: str) -> None:
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
                        'question_id': qa.question_id,
                        'question': qa.question,
                        'answer': qa.answer,
                        'question_type': qa.question_type,
                        'question_date': qa.question_date,
                        'answer_session_ids': qa.answer_session_ids,
                        'is_abstention': qa.is_abstention,
                        'category': qa.category,
                        'category_name': qa.category_name,
                    }
                    for qa in conv.qa_pairs
                ],
                'session_boundaries': conv.session_boundaries,
                'answer_turn_ids': conv.answer_turn_ids,
                'metadata': conv.metadata,
            }
            for conv in conversations
        ],
        'dataset_info': {
            'name': 'LongMemEval',
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
    """CLI for converting LongMemEval dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert LongMemEval dataset to CogCanvas format")
    parser.add_argument(
        "--input", "-i",
        default="experiments/data/longmemeval/data/longmemeval_s_cleaned.json",
        help="Path to LongMemEval JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="experiments/data/longmemeval_converted.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify evidence mapping"
    )

    args = parser.parse_args()

    # Load and convert
    print(f"Loading LongMemEval dataset from {args.input}...")
    raw_data = load_longmemeval(args.input)

    print("\nConverting to evaluation format...")
    conversations = convert_to_eval_format(raw_data)

    # Verify evidence mapping if requested
    if args.verify:
        print("\nVerifying evidence mapping...")
        warnings = 0
        for conv in conversations:
            stats = verify_evidence_mapping(conv)
            status = stats['status']
            if status != 'ok':
                warnings += 1
                print(f"  [{status}] {stats['conversation_id']}: "
                      f"{stats['evidence_turns_found']} evidence turns "
                      f"from {stats['total_evidence_sessions']} evidence sessions "
                      f"(type={stats['question_type']}, abstention={stats['is_abstention']})")
        print(f"\nVerification complete: {warnings} warnings out of {len(conversations)} conversations")

    # Export
    print(f"\nExporting to {args.output}...")
    export_to_json(conversations, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total conversations: {len(conversations)}")
    print(f"Total turns: {sum(len(c.turns) for c in conversations)}")
    print(f"Total QA pairs: {sum(len(c.qa_pairs) for c in conversations)}")
    if conversations:
        print(f"Average turns/conversation: "
              f"{sum(len(c.turns) for c in conversations) / len(conversations):.1f}")
        print(f"Average sessions/conversation: "
              f"{sum(len(c.session_boundaries) for c in conversations) / len(conversations):.1f}")

    # Category distribution
    from collections import defaultdict
    total_by_category = defaultdict(int)
    for conv in conversations:
        for qa in conv.qa_pairs:
            total_by_category[qa.category] += 1

    print("\nQA Category Distribution:")
    for cat in sorted(total_by_category.keys()):
        cat_name = CATEGORY_NAMES.get(cat, f"Category-{cat}")
        print(f"  {cat_name}: {total_by_category[cat]}")

    # Question type distribution
    total_by_type = defaultdict(int)
    for conv in conversations:
        for qa in conv.qa_pairs:
            total_by_type[qa.question_type] += 1

    print("\nQuestion Type Distribution:")
    for qtype in sorted(total_by_type.keys()):
        print(f"  {qtype}: {total_by_type[qtype]}")


if __name__ == "__main__":
    main()
