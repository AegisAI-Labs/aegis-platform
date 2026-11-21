"""Extract an arithmetic expression from a natural-language calculator request.

Used for routing (only send a request to the calculator when there is something to compute)
and to build the calculator input. Returns ``None`` when no expression can be extracted.
"""

import re

CALCULATOR_KEYWORDS = ("calculate", "calculator", "add", "subtract", "multiply", "divide")

_KEYWORD_PATTERN = re.compile(r"\b(" + "|".join(CALCULATOR_KEYWORDS) + r")\b", re.IGNORECASE)
_LEADING_COMMAND = re.compile(r"^\s*(?:please\s+)?(?:calculate|calculator)\b[\s:]*", re.IGNORECASE)
_ARITHMETIC_ONLY = re.compile(r"^[\d\s.+\-*/%()]+$")
_NUMBER = r"(-?\d+(?:\.\d+)?)"

# Simple phrases: "add 2 and 3", "subtract 2 from 5", "multiply 2 by 3", "divide 6 by 2".
_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"\badd\s+{_NUMBER}\s+(?:and|to|plus)\s+{_NUMBER}", re.IGNORECASE), "{0} + {1}"),
    (re.compile(rf"\bsubtract\s+{_NUMBER}\s+from\s+{_NUMBER}", re.IGNORECASE), "{1} - {0}"),
    (
        re.compile(rf"\bmultiply\s+{_NUMBER}\s+(?:by|and|times)\s+{_NUMBER}", re.IGNORECASE),
        "{0} * {1}",
    ),
    (re.compile(rf"\bdivide\s+{_NUMBER}\s+by\s+{_NUMBER}", re.IGNORECASE), "{0} / {1}"),
)


def has_calculator_keyword(message: str) -> bool:
    """Whole-word match, so "address" or "padding" do not count as "add"."""
    return _KEYWORD_PATTERN.search(message) is not None


def extract_expression(message: str) -> str | None:
    for pattern, template in _PHRASES:
        match = pattern.search(message)
        if match:
            return template.format(*match.groups())

    candidate = _LEADING_COMMAND.sub("", message).strip().rstrip("?=").strip()
    if candidate and _ARITHMETIC_ONLY.fullmatch(candidate) and re.search(r"\d", candidate):
        return candidate

    return None
