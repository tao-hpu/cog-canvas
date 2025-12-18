"""Example usage of CogCanvas temporal module."""

from datetime import datetime
from cogcanvas.temporal import (
    TemporalInfo,
    parse_session_datetime,
    normalize_time_expression,
    is_temporal_query,
)

# Example 1: Parse LoCoMo session datetime
print("=" * 70)
print("Example 1: Parsing LoCoMo Session Datetime")
print("=" * 70)

session_str = "1:56 pm on 8 May, 2023"
session_dt = parse_session_datetime(session_str)
print(f"Input:  '{session_str}'")
print(f"Parsed: {session_dt}")
print(f"ISO:    {session_dt.isoformat() if session_dt else 'None'}")
print()

# Example 2: Normalize time expressions
print("=" * 70)
print("Example 2: Normalizing Time Expressions")
print("=" * 70)

reference_date = datetime(2023, 5, 8, 14, 30)
print(f"Reference: {reference_date.strftime('%Y-%m-%d %H:%M (%A)')}")
print()

expressions = [
    "yesterday",
    "last Tuesday",
    "2 weeks ago",
    "May 7, 2023",
]

for expr in expressions:
    info = normalize_time_expression(expr, reference_date)
    print(f"'{expr}' ->")
    print(f"  Normalized:  {info.normalized}")
    print(f"  Type:        {info.temporal_type}")
    print(f"  Confidence:  {info.confidence:.2f}")
    print()

# Example 3: Detect temporal queries
print("=" * 70)
print("Example 3: Detecting Temporal Queries")
print("=" * 70)

queries = [
    "When did Caroline go to the group?",
    "What is her name?",
    "Did she go yesterday?",
    "Tell me about the project status.",
]

for query in queries:
    is_temp, keywords = is_temporal_query(query)
    status = "TEMPORAL" if is_temp else "REGULAR"
    print(f"[{status:8}] {query}")
    if keywords:
        print(f"            Keywords: {', '.join(keywords)}")
print()

# Example 4: Practical use case - filtering based on temporal queries
print("=" * 70)
print("Example 4: Practical Use Case")
print("=" * 70)

query = "What did we decide yesterday?"
is_temp, keywords = is_temporal_query(query)

if is_temp:
    print(f"Query: '{query}'")
    print(f"Detected as temporal query (keywords: {', '.join(keywords)})")
    print()

    # Extract time expression (in real use, would use NER or pattern matching)
    time_expr = "yesterday"
    current_time = datetime(2023, 5, 8, 10, 0)

    info = normalize_time_expression(time_expr, current_time)
    print(f"Time expression: '{time_expr}'")
    print(f"Normalized to:   {info.normalized}")
    print(f"Confidence:      {info.confidence:.2f}")
    print()
    print("Next step: Filter canvas objects by date...")
    print(f"  - Look for objects with turn_id from {info.normalized}")
    print(f"  - Or objects with timestamp matching {info.normalized}")

print("\n" + "=" * 70)
