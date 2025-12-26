"""Track test progress and estimate completion time."""
import re
import sys
from datetime import datetime, timedelta

def track_progress(log_file):
    """Track progress from log file."""
    with open(log_file, 'r') as f:
        content = f.read()

    # Count answered questions
    answered = len(re.findall(r'\s+[✓✗]', content))
    total = 1542

    # Get start time from file modification
    import os
    start_time = datetime.fromtimestamp(os.path.getmtime(log_file))
    elapsed = (datetime.now() - start_time).total_seconds() / 60  # minutes

    if answered > 0:
        rate = answered / elapsed  # questions per minute
        remaining = total - answered
        eta_minutes = remaining / rate if rate > 0 else 0

        print(f"Progress: {answered}/{total} ({answered*100/total:.1f}%)")
        print(f"Elapsed: {elapsed:.1f} minutes")
        print(f"Rate: {rate:.2f} questions/minute")
        print(f"ETA: {eta_minutes:.1f} minutes ({eta_minutes/60:.1f} hours)")
    else:
        print(f"Progress: {answered}/{total} (0.0%)")
        print("Still in compression phase...")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_with_rerank.log"
    track_progress(log_file)
