"""
Gemini batch embedder for the vector backends (memory/vector_backend.py).

Embedding failures must never abort a pipeline run: a failed cache/reuse
lookup is a missed optimisation, not a run failure. embed() therefore
returns None on any failure (network, auth, quota) rather than raising, and
routes the failure through the existing observability event bus so it is
still visible in traces/logs -- see observability/context.py::emit_event.
"""
import logging
from typing import List, Optional

import numpy as np
from google import genai
from google.genai import types

from config.settings import settings
from observability.context import emit_event
from observability.errors import ErrorCode

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _normalize(vector: List[float]) -> List[float]:
    # gemini-embedding-001 only guarantees unit-length output at its native
    # 3072 dimensions; any other output_dimensionality (this project uses
    # 768) must be renormalised so cosine distance behaves as expected.
    arr = np.array(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vector
    return (arr / norm).tolist()


def embed(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch-embed texts. Returns None on failure, [] for an empty input."""
    if not texts:
        return []
    try:
        client = _get_client()
        vectors: List[List[float]] = []
        batch_size = settings.EMBEDDING_BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            response = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(
                    output_dimensionality=settings.EMBEDDING_DIMENSIONS
                ),
            )
            vectors.extend(_normalize(e.values) for e in response.embeddings)
        return vectors
    except Exception as e:
        logger.warning("Embedding request failed, degrading to uncached: %s", e)
        emit_event(
            "embedding_failed",
            message=ErrorCode.PROVIDER_UNAVAILABLE.value,
            detail=str(e),
        )
        return None
