from enum import Enum


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"
    NOT_REACHED = "not_reached"
    SKIPPED_CACHED = "skipped_cached"
