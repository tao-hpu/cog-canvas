"""Diagnose performance issues in the test."""
import re
import sys
from datetime import datetime
from collections import defaultdict

def analyze_log(log_file):
    """Analyze test log for performance issues."""
    print("=" * 60)
    print("Performance Diagnosis Report")
    print("=" * 60)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # 1. Check for rate limiting
    print("\n1. Checking for Rate Limiting...")
    rate_limit_errors = []
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['rate limit', '429', 'too many requests', 'quota']):
            rate_limit_errors.append((i+1, line.strip()))

    if rate_limit_errors:
        print(f"   ⚠️  Found {len(rate_limit_errors)} rate limit errors:")
        for line_num, error in rate_limit_errors[:5]:
            print(f"      Line {line_num}: {error[:100]}")
    else:
        print("   ✅ No rate limiting detected")

    # 2. Check for API errors
    print("\n2. Checking for API Errors...")
    api_errors = []
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['error', 'timeout', 'failed', 'exception']):
            if 'unknown file attribute' not in line.lower():  # Skip unrelated errors
                api_errors.append((i+1, line.strip()))

    if api_errors:
        print(f"   ⚠️  Found {len(api_errors)} errors:")
        for line_num, error in api_errors[:10]:
            print(f"      Line {line_num}: {error[:100]}")
    else:
        print("   ✅ No API errors detected")

    # 3. Analyze phase timing
    print("\n3. Analyzing Phase Timing...")
    compression_starts = []
    answering_starts = []

    for i, line in enumerate(lines):
        if 'Running Rolling Compression' in line:
            compression_starts.append(i)
        elif 'Phase 4: Answering' in line:
            answering_starts.append(i)

    print(f"   Compression phases started: {len(compression_starts)}")
    print(f"   Answering phases started: {len(answering_starts)}")

    # 4. Calculate question answering rate
    print("\n4. Calculating Question Answering Rate...")
    answered_lines = []
    for i, line in enumerate(lines):
        if re.match(r'\s+[✓✗]', line):
            answered_lines.append(i)

    total_answered = len(answered_lines)
    print(f"   Total questions answered: {total_answered} / 1542")

    if total_answered > 10:
        # Estimate speed from first and last answered questions
        first_idx = answered_lines[0]
        last_idx = answered_lines[-1]

        # Try to extract timestamps if available
        # For now, use line numbers as proxy
        questions_answered = len(answered_lines)
        lines_processed = last_idx - first_idx

        print(f"   Questions answered: {questions_answered}")
        print(f"   Lines between first and last: {lines_processed}")

        # Rough estimate: if we know total runtime
        if len(lines) > 100:
            avg_lines_per_question = lines_processed / questions_answered if questions_answered > 0 else 1
            print(f"   Avg lines per question: {avg_lines_per_question:.1f}")

    # 5. Check for retries
    print("\n5. Checking for Retries...")
    retry_count = sum(1 for line in lines if 'retry' in line.lower() or 'waiting' in line.lower())
    if retry_count > 0:
        print(f"   ⚠️  Found {retry_count} retry/waiting messages")
    else:
        print("   ✅ No retries detected")

    # 6. Sample recent output
    print("\n6. Recent Output Sample:")
    print("   " + "-" * 56)
    for line in lines[-10:]:
        print(f"   {line.rstrip()}")
    print("   " + "-" * 56)

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    issues = []
    if rate_limit_errors:
        issues.append(f"Rate limiting detected ({len(rate_limit_errors)} occurrences)")
    if api_errors:
        issues.append(f"API errors detected ({len(api_errors)} occurrences)")
    if retry_count > 10:
        issues.append(f"Many retries detected ({retry_count} occurrences)")

    if issues:
        print("⚠️  Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No obvious performance issues detected")
        print("   Slowness may be due to:")
        print("   - LLM API latency (inherent)")
        print("   - Graph expansion overhead")
        print("   - Extraction overhead")

    print("\nProgress: {}/{} ({:.1f}%)".format(
        total_answered, 1542, total_answered * 100 / 1542 if total_answered > 0 else 0
    ))

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude/-Users-TaoTao-Desktop-Learn------cog-canvas-all/tasks/b41a046.output"
    analyze_log(log_file)
