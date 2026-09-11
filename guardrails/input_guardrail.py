"""
Input guardrail: reject a research topic before any LLM call runs. Deliberately
heuristic (regex + simple ratios), not a model call -- this must be cheap and
must never itself require a provider that could be down.
"""
import re

MIN_LENGTH = 3
MAX_LENGTH = 300


class GuardrailBlockedError(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


_INJECTION_PATTERNS = [
    (re.compile(r"ignore (all |any |the )?(previous|prior|above|earlier) instructions", re.I),
     "prompt injection attempt"),
    (re.compile(r"\byou are now\b|\bact as\b|\bpretend (to be|you are)\b", re.I),
     "role-override attempt"),
    (re.compile(r"\bsystem prompt\b|\bjailbreak\b", re.I),
     "system-prompt extraction attempt"),
]
_CODE_GEN_PATTERN = re.compile(
    r"\bwrite (me )?(a |some )?(python|javascript|java|c\+\+|code|script|program|function)\b", re.I
)
_PII_PATTERN = re.compile(
    r"\b(home address|phone number|social security number|ssn|date of birth|bank account)\b", re.I
)


def validate_topic(topic: str) -> None:
    """Raises GuardrailBlockedError with a specific reason, or returns None."""
    stripped = topic.strip()

    if len(stripped) < MIN_LENGTH:
        raise GuardrailBlockedError(f"Topic is too short (minimum {MIN_LENGTH} characters).")
    if len(stripped) > MAX_LENGTH:
        raise GuardrailBlockedError(f"Topic is too long (maximum {MAX_LENGTH} characters).")

    letters = sum(c.isalpha() for c in stripped)
    if letters < len(stripped) * 0.3:
        raise GuardrailBlockedError("Topic does not look like natural-language text.")

    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            raise GuardrailBlockedError(f"Topic rejected: {reason}.")

    if _CODE_GEN_PATTERN.search(stripped):
        raise GuardrailBlockedError(
            "Topic rejected: this looks like a code-generation request, not a research topic."
        )

    if _PII_PATTERN.search(stripped):
        raise GuardrailBlockedError(
            "Topic rejected: appears to request personal information about a private individual."
        )
