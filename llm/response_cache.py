"""
Exact-match, in-process response cache for deterministic, low-temperature
pipeline stages (classifier, evidence_comparator). RouterLLM gates which
agent aliases are cacheable; the Writer is never cached (see
RouterLLM._CACHEABLE_ALIASES).

Module-level (not per-RouterLLM-instance) because a fresh RouterLLM is
built on every stage call (see agents/*.py's get_llm() usage) -- an
instance-level cache would never see a second call to hit against.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

_store: Dict[str, "CachedResponse"] = {}


@dataclass
class CachedResponse:
    result: Any
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: Optional[float]
    latency_ms: float


def make_key(alias: str, messages: Any) -> str:
    digest = hashlib.sha256(repr(messages).encode("utf-8", "replace")).hexdigest()
    return f"{alias}:{digest}"


def get(key: str) -> Optional[CachedResponse]:
    return _store.get(key)


def set(key: str, entry: CachedResponse) -> None:
    _store[key] = entry


def reset_for_tests() -> None:
    _store.clear()
