"""
Temporal Enhancement Module for CogCanvas.

This module provides enhanced temporal information extraction and retrieval
to improve performance on temporal reasoning questions.

Key Features:
1. Regex-based date extraction (preserves verbatim dates)
2. Relative time normalization (when session time is available)
3. Temporal artifact creation (dedicated time-stamped facts)
4. Time-aware retrieval boosting
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

from cogcanvas.models import CanvasObject, ObjectType


# =============================================================================
# Date Patterns (for verbatim extraction)
# =============================================================================

DATE_PATTERNS = [
    # Full dates: "7 May 2023", "May 7, 2023", "7th May 2023"
    r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
    r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b',

    # Month + Year: "June 2023", "May 2022"
    r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',

    # Year only: "2022", "2023"
    r'\b(20\d{2})\b',

    # ISO format: "2023-05-07"
    r'\b(\d{4}-\d{2}-\d{2})\b',

    # Relative with specifics: "the week before 9 June 2023", "the sunday before 25 May"
    r'\b((?:the\s+)?(?:week|day|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\s+(?:before|after)\s+\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{0,4})\b',
]

RELATIVE_TIME_PATTERNS = [
    # Days
    r'\b(yesterday|today|tomorrow)\b',
    r'\b(last\s+(?:night|week|month|year|sunday|monday|tuesday|wednesday|thursday|friday|saturday))\b',
    r'\b(next\s+(?:week|month|year|sunday|monday|tuesday|wednesday|thursday|friday|saturday))\b',
    r'\b(this\s+(?:week|month|year|morning|afternoon|evening))\b',
    r'\b(\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b',
    r'\b(in\s+\d+\s+(?:days?|weeks?|months?|years?))\b',
]

# Compile patterns for efficiency
COMPILED_DATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]
COMPILED_RELATIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RELATIVE_TIME_PATTERNS]


# =============================================================================
# Relative → Absolute Time Conversion
# =============================================================================

def parse_session_datetime(session_datetime: str) -> Optional[datetime]:
    """
    Parse session datetime string to datetime object.

    Formats supported:
    - "1:56 pm on 8 May, 2023"
    - "10:37 am on 27 June, 2023"

    Args:
        session_datetime: Session datetime string from LoCoMo

    Returns:
        datetime object or None if parsing fails
    """
    if not session_datetime:
        return None

    # Pattern: "1:56 pm on 8 May, 2023"
    pattern = r'(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})'
    match = re.match(pattern, session_datetime, re.IGNORECASE)

    if match:
        hour, minute, ampm, day, month_str, year = match.groups()
        hour = int(hour)
        minute = int(minute)
        day = int(day)
        year = int(year)

        # Convert to 24-hour
        if ampm.lower() == 'pm' and hour != 12:
            hour += 12
        elif ampm.lower() == 'am' and hour == 12:
            hour = 0

        # Parse month
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month = months.get(month_str.lower(), 1)

        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    return None


def resolve_relative_time(
    relative_expr: str,
    session_datetime: str,
) -> Optional[str]:
    """
    Convert relative time expression to absolute date.

    Args:
        relative_expr: e.g., "yesterday", "last week", "3 days ago"
        session_datetime: e.g., "1:56 pm on 8 May, 2023"

    Returns:
        Absolute date string e.g., "7 May 2023" or None if can't resolve
    """
    base_date = parse_session_datetime(session_datetime)
    if not base_date:
        return None

    expr_lower = relative_expr.lower().strip()
    result_date = None

    # Yesterday/today/tomorrow
    if expr_lower == 'yesterday':
        result_date = base_date - timedelta(days=1)
    elif expr_lower == 'today':
        result_date = base_date
    elif expr_lower == 'tomorrow':
        result_date = base_date + timedelta(days=1)

    # Last week/month/year
    elif expr_lower == 'last week':
        result_date = base_date - timedelta(weeks=1)
    elif expr_lower == 'last month':
        result_date = base_date - timedelta(days=30)
    elif expr_lower == 'last year':
        result_date = base_date - timedelta(days=365)

    # Next week/month
    elif expr_lower == 'next week':
        result_date = base_date + timedelta(weeks=1)
    elif expr_lower == 'next month':
        result_date = base_date + timedelta(days=30)

    # N days/weeks/months ago
    ago_match = re.match(r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago', expr_lower)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2).rstrip('s')
        if unit == 'day':
            result_date = base_date - timedelta(days=num)
        elif unit == 'week':
            result_date = base_date - timedelta(weeks=num)
        elif unit == 'month':
            result_date = base_date - timedelta(days=num * 30)
        elif unit == 'year':
            result_date = base_date - timedelta(days=num * 365)

    # Last Sunday/Monday/etc
    weekday_match = re.match(r'last\s+(sunday|monday|tuesday|wednesday|thursday|friday|saturday)', expr_lower)
    if weekday_match:
        weekdays = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}
        target_weekday = weekdays[weekday_match.group(1)]
        current_weekday = base_date.weekday()
        days_back = (current_weekday - target_weekday) % 7
        if days_back == 0:
            days_back = 7
        result_date = base_date - timedelta(days=days_back)

    # Format result
    if result_date:
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        return f"{result_date.day} {months[result_date.month - 1]} {result_date.year}"

    return None


@dataclass
class TemporalExtraction:
    """A temporal expression extracted from text."""
    expression: str          # The verbatim time expression
    start_pos: int          # Start position in text
    end_pos: int            # End position in text
    is_absolute: bool       # True if it's an absolute date (e.g., "May 7, 2023")
    context: str            # Surrounding text for context


class TemporalEnhancer:
    """
    Enhances temporal information extraction and retrieval.

    This class:
    1. Extracts date/time expressions using regex (preserves verbatim)
    2. Creates dedicated temporal artifacts
    3. Boosts retrieval for temporal queries
    """

    def __init__(self, context_window: int = 100):
        """
        Initialize the temporal enhancer.

        Args:
            context_window: Characters of context to capture around dates
        """
        self.context_window = context_window

    def extract_temporal_expressions(self, text: str) -> List[TemporalExtraction]:
        """
        Extract all temporal expressions from text.

        Args:
            text: Text to extract from

        Returns:
            List of temporal extractions with positions and context
        """
        all_matches = []

        # Collect all matches from all patterns
        for pattern in COMPILED_DATE_PATTERNS:
            for match in pattern.finditer(text):
                all_matches.append((match.start(), match.end(), match.group(0), True))

        for pattern in COMPILED_RELATIVE_PATTERNS:
            for match in pattern.finditer(text):
                all_matches.append((match.start(), match.end(), match.group(0), False))

        # Sort by length descending, then by position
        all_matches.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

        # Keep only the longest non-overlapping matches
        used_ranges = []
        extractions = []

        for start, end, expr, is_abs in all_matches:
            # Check if this range overlaps with any used range
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break

            if not overlaps:
                used_ranges.append((start, end))
                extractions.append(TemporalExtraction(
                    expression=expr,
                    start_pos=start,
                    end_pos=end,
                    is_absolute=is_abs,
                    context=self._get_context(text, start, end),
                ))

        # Sort by position
        extractions.sort(key=lambda x: x.start_pos)
        return extractions

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - self.context_window)
        ctx_end = min(len(text), end + self.context_window)
        return text[ctx_start:ctx_end].strip()

    def create_temporal_artifacts(
        self,
        text: str,
        turn_id: int,
        source: str = "user",
        session_datetime: Optional[str] = None,
    ) -> List[CanvasObject]:
        """
        Create dedicated temporal artifacts from text.

        These artifacts preserve the exact date strings and associate
        them with their context for better retrieval.

        When session_datetime is provided, relative times (yesterday, last week)
        are converted to absolute dates for better matching with ground truth.

        Args:
            text: Text to extract from
            turn_id: Current conversation turn
            source: "user" or "assistant"
            session_datetime: Session timestamp for relative time resolution

        Returns:
            List of temporal CanvasObjects
        """
        extractions = self.extract_temporal_expressions(text)
        artifacts = []

        for ext in extractions:
            # Try to resolve relative time to absolute
            resolved_time = None
            if not ext.is_absolute and session_datetime:
                resolved_time = resolve_relative_time(ext.expression, session_datetime)

            # Create content with both raw and resolved time
            if resolved_time:
                content = f"TIME: {resolved_time} (originally: {ext.expression})"
                event_time = resolved_time  # Use resolved absolute time
            else:
                content = f"TIME: {ext.expression}"
                event_time = ext.expression if ext.is_absolute else None

            if ext.context and len(ext.context) > len(ext.expression) + 10:
                # Add context if meaningful
                content = f"{content} - {ext.context}"

            artifact = CanvasObject(
                type=ObjectType.EVENT,  # Use EVENT type for temporal artifacts
                content=content,
                quote=ext.expression,  # Verbatim preservation
                context=f"Temporal extraction from turn {turn_id}",
                source=source,
                turn_id=turn_id,
                event_time=event_time,  # Normalized/resolved time
                event_time_raw=ext.expression,  # Original expression
                session_datetime=session_datetime,
                confidence=0.95 if ext.is_absolute else (0.9 if resolved_time else 0.85),
            )
            artifacts.append(artifact)

        return artifacts

    def is_temporal_query(self, query: str) -> bool:
        """
        Check if a query is asking about time/date.

        Args:
            query: The query string

        Returns:
            True if query is temporal
        """
        temporal_keywords = [
            'when', 'what time', 'what date', 'which day', 'which month',
            'which year', 'how long ago', 'how recently', 'before', 'after',
            'during', 'since', 'until', 'schedule', 'deadline', 'appointment',
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in temporal_keywords)

    def boost_temporal_results(
        self,
        objects: List[CanvasObject],
        scores: List[float],
        query: str,
        boost_factor: float = 1.5,
    ) -> Tuple[List[CanvasObject], List[float]]:
        """
        Boost scores for objects with temporal information when query is temporal.

        Args:
            objects: Retrieved objects
            scores: Their retrieval scores
            query: The query
            boost_factor: How much to boost temporal objects

        Returns:
            Re-ranked objects and scores
        """
        if not self.is_temporal_query(query):
            return objects, scores

        # Boost objects with temporal info
        boosted_scores = []
        for obj, score in zip(objects, scores):
            if obj.event_time_raw or (obj.content and obj.content.startswith("TIME:")):
                boosted_scores.append(score * boost_factor)
            else:
                boosted_scores.append(score)

        # Re-sort by boosted scores
        paired = list(zip(objects, boosted_scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        return [p[0] for p in paired], [p[1] for p in paired]


def enhance_extraction_prompt(base_prompt: str) -> str:
    """
    Enhance the extraction prompt to better capture temporal information.

    Args:
        base_prompt: The original extraction prompt

    Returns:
        Enhanced prompt with temporal emphasis
    """
    temporal_emphasis = """

CRITICAL TEMPORAL EXTRACTION RULES:
1. When you see ANY date or time expression, you MUST preserve it EXACTLY as written
2. DO NOT convert "7 May 2023" to "recently" or "last week"
3. DO NOT convert "June 2023" to "this summer" or "a few months ago"
4. If the text says "the sunday before 25 May 2023", extract EXACTLY that string
5. Create a separate KEY_FACT for each date/time with format: "EVENT happened on DATE"

Example of CORRECT temporal extraction:
Text: "Caroline went to the LGBTQ support group on 7 May 2023"
Extract: {"type": "key_fact", "content": "Caroline went to LGBTQ support group on 7 May 2023",
          "citation": "went to the LGBTQ support group on 7 May 2023",
          "time_expression": "7 May 2023"}

Example of WRONG temporal extraction (DO NOT DO THIS):
Text: "Caroline went to the LGBTQ support group on 7 May 2023"
Extract: {"content": "Caroline went to LGBTQ support group recently"} ← WRONG! Date lost!
"""
    return base_prompt + temporal_emphasis


# =============================================================================
# Integration helpers
# =============================================================================

def extract_and_merge_temporal(
    text: str,
    llm_objects: List[CanvasObject],
    turn_id: int,
    source: str = "user",
    session_datetime: Optional[str] = None,
) -> List[CanvasObject]:
    """
    Extract temporal artifacts and merge with LLM-extracted objects.

    This ensures we capture dates even if LLM misses them.
    When session_datetime is provided, relative times are converted to absolute.

    Args:
        text: Original text
        llm_objects: Objects extracted by LLM
        turn_id: Current turn
        source: Message source
        session_datetime: Session timestamp for relative time resolution (e.g., "1:56 pm on 8 May, 2023")

    Returns:
        Merged list of objects
    """
    enhancer = TemporalEnhancer()

    # Get temporal artifacts from regex (with relative time resolution)
    temporal_artifacts = enhancer.create_temporal_artifacts(
        text, turn_id, source, session_datetime=session_datetime
    )

    # Check which dates are already captured by LLM
    existing_dates = set()
    for obj in llm_objects:
        if obj.event_time_raw:
            existing_dates.add(obj.event_time_raw.lower())
        if obj.event_time:
            existing_dates.add(obj.event_time.lower())

    # Add only new temporal artifacts
    new_artifacts = []
    for artifact in temporal_artifacts:
        raw_lower = artifact.event_time_raw.lower() if artifact.event_time_raw else ""
        resolved_lower = artifact.event_time.lower() if artifact.event_time else ""

        # Skip if already captured (either raw or resolved)
        if raw_lower not in existing_dates and resolved_lower not in existing_dates:
            new_artifacts.append(artifact)

    # Also update LLM objects with resolved times if they have relative expressions
    if session_datetime:
        for obj in llm_objects:
            if obj.event_time_raw and not obj.event_time:
                resolved = resolve_relative_time(obj.event_time_raw, session_datetime)
                if resolved:
                    obj.event_time = resolved

    return llm_objects + new_artifacts
