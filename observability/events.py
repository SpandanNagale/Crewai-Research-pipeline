from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from observability.node_state import NodeState


class RunEvent(BaseModel):
    run_id: str
    seq: int
    timestamp: datetime
    stage: str
    event_type: str
    node_state: Optional[NodeState] = None
    model_alias: Optional[str] = None
    resolved_deployment: Optional[str] = None
    attempt: Optional[int] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cache_hit: Optional[bool] = None
    tokens_saved: Optional[int] = None
    cost_avoided_usd: Optional[float] = None
    seconds_saved: Optional[float] = None
    message: str = ""
    detail: Optional[str] = None
