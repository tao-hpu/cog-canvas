"""
Debug retrieval failure for a specific case.
Analyze why CogCanvas fails to retrieve relevant information.
"""

import json
from pathlib import Path

def analyze_case(
    cogcanvas_result_file,
    conversation_id,
    question_idx
):
    """
    Analyze a specific failure case.

    Args:
        cogcanvas_result_file: Path to CogCanvas result JSON
        conversation_id: ID of the conversation
        question_idx: Index of the question
    """
    # Load CogCanvas results
    with open(cogcanvas_result_file) as f:
        data = json.load(f)

    # Find the specific conversation
    conv = None
    for c in data['conversations']:
        if c['id'] == conversation_id:
            conv = c
            break

    if not conv:
        print(f"Conversation {conversation_id} not found")
        return

    # Find the specific question
    if question_idx >= len(conv['questions']):
        print(f"Question index {question_idx} out of range")
        return

    question_data = conv['questions'][question_idx]

    print(f"\n{'='*80}")
    print(f"Debugging Case: {conversation_id} - Question {question_idx}")
    print(f"{'='*80}")
    print(f"\nQuestion:")
    print(f"  {question_data['question']}")
    print(f"\nGround Truth:")
    print(f"  {question_data['ground_truth']}")
    print(f"\nCogCanvas Answer:")
    print(f"  {question_data['answer'][:300]}...")
    print(f"\nPassed: {question_data.get('passed', False)}")

    # Check if there's retrieval info in the question data
    if 'retrieval_info' in question_data:
        ret_info = question_data['retrieval_info']
        print(f"\n{'='*80}")
        print("Retrieval Information")
        print(f"{'='*80}")
        print(f"Number of objects retrieved: {len(ret_info.get('objects', []))}")
        print(f"Retrieval scores: {ret_info.get('scores', [])[:5]}")

        print(f"\nTop 5 Retrieved Objects:")
        for i, obj in enumerate(ret_info.get('objects', [])[:5], 1):
            score = ret_info.get('scores', [])[i-1] if i-1 < len(ret_info.get('scores', [])) else 'N/A'
            print(f"\n  Object {i} (Score: {score:.3f}):")
            print(f"    Type: {obj.get('type', 'N/A')}")
            print(f"    Content: {obj.get('content', '')[:100]}...")
            if obj.get('quote'):
                print(f"    Quote: {obj.get('quote', '')[:100]}...")
    else:
        print(f"\nNo retrieval_info available in question data")

    # Check if we have canvas state
    if 'canvas_state' in question_data:
        canvas = question_data['canvas_state']
        print(f"\n{'='*80}")
        print("Canvas State")
        print(f"{'='*80}")
        print(f"Total objects in canvas: {len(canvas.get('objects', []))}")

        # Search for objects containing keywords from ground truth
        keywords = question_data['ground_truth'].lower().split()
        matching_objects = []

        for obj in canvas.get('objects', []):
            content = obj.get('content', '').lower()
            quote = obj.get('quote', '').lower()
            combined = f"{content} {quote}"

            if any(kw in combined for kw in keywords):
                matching_objects.append(obj)

        print(f"\nObjects containing ground truth keywords: {len(matching_objects)}")
        for i, obj in enumerate(matching_objects[:3], 1):
            print(f"\n  Match {i}:")
            print(f"    Type: {obj.get('type', 'N/A')}")
            print(f"    Content: {obj.get('content', '')[:100]}...")
            print(f"    Quote: {obj.get('quote', '')[:100]}...")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    base_dir = Path(__file__).parent / "results"

    # Analyze Case 7: James visited Greenland
    print("\n" + "="*80)
    print("CASE 7: James visited Greenland during Canada trip")
    print("="*80)
    analyze_case(
        base_dir / "rolling_locomo_cogcanvas.json",
        "locomo_006",
        35
    )

    # Analyze Case 8: Jolene's pendant from France
    print("\n" + "="*80)
    print("CASE 8: Jolene's pendant from France")
    print("="*80)
    analyze_case(
        base_dir / "rolling_locomo_cogcanvas.json",
        "locomo_007",
        5
    )
