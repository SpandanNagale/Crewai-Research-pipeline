from enum import Enum
from typing import Dict, Tuple


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ErrorCode(str, Enum):
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_AUTH_FAILED = "PROVIDER_AUTH_FAILED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    TOOL_FETCH_FAILED = "TOOL_FETCH_FAILED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    GROUNDEDNESS_BELOW_THRESHOLD = "GROUNDEDNESS_BELOW_THRESHOLD"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INTERNAL = "INTERNAL"


ERROR_SENTENCES: Dict[ErrorCode, Tuple[str, Severity]] = {
    ErrorCode.PROVIDER_TIMEOUT: ("The AI provider took too long to respond.", Severity.WARNING),
    ErrorCode.PROVIDER_RATE_LIMIT: ("The AI provider's rate limit was hit.", Severity.WARNING),
    ErrorCode.PROVIDER_AUTH_FAILED: ("The AI provider rejected our credentials.", Severity.ERROR),
    ErrorCode.PROVIDER_UNAVAILABLE: ("Every configured provider for this stage failed.", Severity.ERROR),
    ErrorCode.SCHEMA_VALIDATION_FAILED: ("The model's response didn't match the expected format.", Severity.WARNING),
    ErrorCode.GUARDRAIL_BLOCKED: ("A safety guardrail blocked this request.", Severity.ERROR),
    ErrorCode.TOOL_FETCH_FAILED: ("A research tool call failed.", Severity.WARNING),
    ErrorCode.SEARCH_EXHAUSTED: ("No further search results were available.", Severity.WARNING),
    ErrorCode.GROUNDEDNESS_BELOW_THRESHOLD: ("The report didn't meet the groundedness threshold.", Severity.WARNING),
    ErrorCode.BUDGET_EXCEEDED: ("The run exceeded its cost/token budget.", Severity.ERROR),
    ErrorCode.INTERNAL: ("An internal error occurred.", Severity.ERROR),
}


def sentence_for(code: ErrorCode) -> str:
    return ERROR_SENTENCES[code][0]


def severity_for(code: ErrorCode) -> Severity:
    return ERROR_SENTENCES[code][1]
