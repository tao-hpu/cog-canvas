"""
DuRecDial 2.0 Dataset Adapter for CogCanvas Evaluation.

Adapts the DuRecDial 2.0 dataset to the CogCanvas evaluation format (LoCoMo-compatible).

DuRecDial 2.0 Characteristics:
- Goal-driven dialogue (Recommendation)
- Contains Knowledge, Goal, and Conversation fields
- We simulate QA pairs for evaluation:
  1. Single-hop: User profile constraints (e.g., preferred genre)
  2. Multi-hop: Knowledge attributes of recommended items
  3. Temporal: Preference shifts or rejection reasons

"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from experiments.locomo_adapter import LoCoMoConversation, LoCoMoQAPair
from experiments.data_gen import ConversationTurn

# =============================================================================
# Configuration
# =============================================================================

MIN_TURNS_DEFAULT = 30  # Default filter for "long" conversations
COMPRESSION_RATIO = 0.5  # Compress at 50% of the conversation

# =============================================================================
# QA Generation Heuristics
# =============================================================================

def generate_single_hop_questions(goal: Dict[str, Any]) -> List[LoCoMoQAPair]:
    """Generate single-hop questions based on user profile/constraints."""
    qa_pairs = []
    
    # Example goal structure from user description (adapting to what we saw in search/sample)
    # We expect something like goal['user_profile'] or goal constraints
    
    profile = goal.get('user_profile', {})
    if not profile and isinstance(goal, list):
         # Sometimes goal is a list of subgoals
         pass 

    # If profile is a dict, iterate keys
    if isinstance(profile, dict):
        for key, value in profile.items():
            if isinstance(value, list):
                val_str = ", ".join(value)
            else:
                val_str = str(value)
            
            # Skip empty or technical keys
            if not val_str or key in ['user_id', 'status']:
                continue
                
            question = f"What {key} does the user prefer?"
            qa_pairs.append(LoCoMoQAPair(
                question=question,
                answer=val_str,
                evidence=[], # DuRecDial doesn't have per-sentence IDs usually, we'll try to map later if possible
                category=1 # Single-hop
            ))
            
    return qa_pairs

def generate_multi_hop_questions(goal: Any, knowledge: List[Any]) -> List[LoCoMoQAPair]:
    """Generate multi-hop questions about recommended items."""
    qa_pairs = []
    
    # Identify the final recommendation
    final_rec = None
    if isinstance(goal, dict):
        final_rec = goal.get('final_recommendation')
    
    if not final_rec:
        # Try to infer from goal string or knowledge? 
        # DuRecDial goals are strings like "...-->[3] Movie recommendation(The Equation of Love & Death)..."
        # We can parse this string to find the recommendation
        pass

    # Basic parsing of goal string if final_rec is missing
    # Handle if goal is passed as string directly or inside dict
    goal_str = goal if isinstance(goal, str) else goal.get('goal_string', "") if isinstance(goal, dict) else ""
    
    if not final_rec and goal_str:
        import re
        # Look for "recommendation(Name)" pattern
        # The pattern in sample is "Movie recommendation(Name)" or similar
        match = re.search(r"recommendation\((.*?)\)", goal_str)
        if match:
            final_rec = match.group(1)

    # If still no final_rec, try to find it in goal_type_list or similar if available passed in goal dict
    # But for now let's rely on the string parsing above which covers most DuRecDial cases

    if final_rec:
        # Normalize final_rec for matching
        final_rec_norm = final_rec.lower().strip()

        # Find attributes in knowledge
        # Knowledge can be list of dicts OR list of lists [sub, pred, obj]
        rec_knowledge = []
        for k in knowledge:
            if isinstance(k, dict):
                if k.get('entity') == final_rec:
                    rec_knowledge.append(k)
            elif isinstance(k, list) and len(k) >= 3:
                # [Entity, Attr, Value]
                if k[0] and k[0].lower().strip() == final_rec_norm:
                    rec_knowledge.append({'entity': k[0], 'attr': k[1], 'value': k[2]})
        
        for k in rec_knowledge:
            attr = k.get('attr') or k.get('attribute')
            val = k.get('value') or k.get('val')
            
            if attr and val:
                question = f"What is the {attr} of the recommended item '{final_rec}'?"
                qa_pairs.append(LoCoMoQAPair(
                    question=question,
                    answer=val,
                    evidence=[], 
                    category=3 # Multi-hop (requires linking rec -> knowledge)
                ))
                
    return qa_pairs

def generate_temporal_questions(turns: List[Any]) -> List[LoCoMoQAPair]:
    """Generate temporal questions based on flow (e.g., rejections)."""
    qa_pairs = []
    
    # Simple heuristic: Look for "no", "don't like", "seen it" in user turns
    # Then ask about the previous assistant turn's suggestion
    
    for i, turn in enumerate(turns):
        role = ""
        content = ""
        
        if isinstance(turn, str):
            # Assume alternating User starts
            role = 'user' if i % 2 == 0 else 'assistant'
            content = turn
        elif isinstance(turn, dict):
            role = turn.get('role') or turn.get('speaker')
            content = turn.get('content') or turn.get('utterance')

        if role in ['user', 'User']:
            text = str(content).lower()
            if "seen that" in text or "seen it" in text or "watched it" in text:
                # Likely a rejection of previous recommendation
                # Look at previous turn (i-1) if it exists and is assistant
                prev_is_assistant = False
                if i > 0:
                    prev_turn = turns[i-1]
                    if isinstance(prev_turn, str):
                        prev_is_assistant = ((i-1) % 2 != 0)
                    elif isinstance(prev_turn, dict):
                        prev_role = prev_turn.get('role') or prev_turn.get('speaker')
                        prev_is_assistant = (prev_role not in ['user', 'User'])
                
                if prev_is_assistant:
                    # Try to extract the movie name from previous turn? 
                    # For now, generic question
                    question = "Why did the user reject the recommendation made at turn {}".format(i) # 1-based index handled later
                    answer = "The user had already seen it."
                    qa_pairs.append(LoCoMoQAPair(
                        question=question,
                        answer=answer,
                        evidence=[],
                        category=2 # Temporal
                    ))
            elif "prefer" in text or "actually" in text or "how about" in text:
                # Preference change
                question = "At what point did the user change their preference?"
                answer = f"Turn {i+1}" # 1-based
                qa_pairs.append(LoCoMoQAPair(
                    question=question,
                    answer=answer,
                    evidence=[],
                    category=2
                ))
                
    return qa_pairs

# =============================================================================
# Conversion Logic
# =============================================================================

def convert_durecdial_to_locomo(
    durecdial_data: List[Dict[str, Any]], 
    min_turns: int = 0,
    id_prefix: str = "drd"
) -> List[LoCoMoConversation]:
    """
    Convert DuRecDial data to LoCoMo format.
    
    Args:
        durecdial_data: List of raw conversation dicts
        min_turns: Filter for minimum number of turns
        id_prefix: Prefix for conversation IDs (default: "drd")
        
    Returns:
        List of LoCoMoConversation objects
    """
    conversations = []
    
    for idx, item in enumerate(durecdial_data):
        raw_turns = item.get('conversation', []) or item.get('turns', [])
        
        # Check length (turns)
        if len(raw_turns) < min_turns:
            continue
            
        # Build turns and dialogue mapping
        turns = []
        dialogue_id_to_turn = {}
        
        # Determine roles. Usually 'user' and 'assistant'/'system'.
        # We need to ensure alternating or merge same-speaker turns
        current_user_text = ""
        current_assistant_text = ""
        turn_id = 1
        
        # Helper to flush a turn
        def flush_turn(u, a, t_id):
            return ConversationTurn(
                turn_id=t_id,
                user=u,
                assistant=a,
                session_datetime=None
            )

        # Iterate and merge
        # Strategy: A turn object in CogCanvas is (User, Assistant).
        # So we collect User text, then Assistant text, then emit.
        
        temp_user = []
        temp_assistant = []
        
        for i, t in enumerate(raw_turns):
            if isinstance(t, str):
                # Simple string list case (DuRecDial raw)
                # Assume alternating: 0=User, 1=Assistant, ...
                role = 'user' if i % 2 == 0 else 'assistant'
                content = t
            else:
                # Dict case (if converted or other format)
                role = t.get('role') or t.get('speaker')
                content = t.get('content') or t.get('utterance')
            
            if role in ['user', 'User']:
                # If we have pending assistant text, that closes the previous turn
                if temp_assistant:
                    turns.append(flush_turn("\n".join(temp_user) if temp_user else "[Start]", "\n".join(temp_assistant), turn_id))
                    turn_id += 1
                    temp_user = []
                    temp_assistant = []
                
                temp_user.append(content)
                
            else: # assistant
                temp_assistant.append(content)
        
        # Flush final turn
        if temp_user or temp_assistant:
            turns.append(flush_turn(
                "\n".join(temp_user) if temp_user else "[Continued]", 
                "\n".join(temp_assistant) if temp_assistant else "[End]", 
                turn_id
            ))

        # Generate QA
        # DuRecDial: goal is string, user_profile is dict
        raw_goal = item.get('goal', "")
        user_profile = item.get('user_profile', {})
        knowledge = item.get('knowledge', [])
        
        # Construct a composite goal object for generators
        goal_context = {
            'final_recommendation': None, # Will be parsed from raw_goal string in multi-hop gen
            'user_profile': user_profile,
            'goal_string': raw_goal
        }
        
        # Pass the raw_goal string as the "goal" for multi-hop generator to parse
        qa_pairs = []
        qa_pairs.extend(generate_single_hop_questions(goal_context))
        # Pass raw_goal string directly to multi-hop generator which now expects it or parses it
        qa_pairs.extend(generate_multi_hop_questions(raw_goal, knowledge))
        qa_pairs.extend(generate_temporal_questions(raw_turns))
        
        # Create Metadata
        metadata = {
            'source': 'DuRecDial2.0',
            'original_id': item.get('conversation_id', f'{id_prefix}_{idx}'),
            'goal': raw_goal
        }
        
        # Create Object
        conv_id = f"{id_prefix}_{idx:04d}"
        
        # Map simulated evidence (not really used but required by type)
        # We'll just map "turn_X" to X
        dialogue_id_to_turn = {f"turn_{t.turn_id}": t.turn_id for t in turns}
        
        conversations.append(LoCoMoConversation(
            id=conv_id,
            speaker_a="User",
            speaker_b="Assistant",
            turns=turns,
            qa_pairs=qa_pairs,
            dialogue_id_to_turn=dialogue_id_to_turn,
            metadata=metadata
        ))
        
    return conversations
def load_durecdial_file(path: str) -> List[Dict[str, Any]]:
    """Load DuRecDial from JSONL or JSON."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        # Check if it's a list (JSON) or JSONL
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            data = json.load(f)
        else:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

# =============================================================================
# Main / CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DuRecDial Adapter Test")
    parser.add_argument("--input", default="experiments/data/durecdial_sample.jsonl")
    parser.add_argument("--min-turns", type=int, default=0) # 0 for test
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    raw = load_durecdial_file(args.input)
    print(f"Loaded {len(raw)} items.")
    
    convs = convert_durecdial_to_locomo(raw, min_turns=args.min_turns)
    print(f"Converted {len(convs)} conversations.")
    
    if convs:
        c = convs[0]
        print(f"\nSample Conversation ({c.id}):")
        print(f"Turns: {len(c.turns)}")
        print(f"QA Pairs: {len(c.qa_pairs)}")
        for qa in c.qa_pairs:
            print(f" - [{qa.category_name}] Q: {qa.question} | A: {qa.answer}")
