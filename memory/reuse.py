"""
Knowledge reuse: before running a full pipeline for a new question, embed
the topic and compare it against past research_questions. If a close-enough
match exists, surface its existing findings/conclusions instead of (or in
addition to) re-running full research.

Also maintains a second, finer-grained index over individual findings and
conclusions for the GET /knowledge/search semantic-search endpoint.

Written entirely against memory.vector_store.ResearchMemory's public
interface -- never against a backend-specific client -- so the storage
engine can change without touching this file.
"""
import hashlib
import logging
from typing import List, Optional

from config.settings import settings
from db import repository
from memory.vector_store import ResearchMemory

logger = logging.getLogger(__name__)

QUESTIONS_COLLECTION = "research_questions_index"
KNOWLEDGE_COLLECTION = "knowledge_index"
SUBQUESTIONS_COLLECTION = "sub_questions_index"


def _make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class KnowledgeReuseChecker:
    def __init__(self):
        self._questions_memory = ResearchMemory(collection_name=QUESTIONS_COLLECTION)
        self._knowledge_memory = ResearchMemory(collection_name=KNOWLEDGE_COLLECTION)
        self._subquestions_memory = ResearchMemory(collection_name=SUBQUESTIONS_COLLECTION)

    def find_similar(self, topic: str) -> Optional[dict]:
        """
        Return the full trace dict (repository.get_trace) for the closest past
        question if its similarity to `topic` is >= KNOWLEDGE_REUSE_THRESHOLD,
        else None.
        """
        try:
            hits = self._questions_memory.search(topic, n_results=1)
        except Exception as e:
            logger.warning("Knowledge reuse similarity search failed: %s", e)
            return None

        if not hits:
            return None

        best = hits[0]
        if best.similarity < settings.KNOWLEDGE_REUSE_THRESHOLD:
            return None

        question_id = int(best.metadata["question_id"])
        trace = repository.get_trace(question_id)
        if trace is not None:
            trace["similarity"] = round(best.similarity, 4)
        return trace

    def find_similar_subquestion(self, sub_question: str) -> Optional[dict]:
        """
        Partial-reuse counterpart to find_similar(): matches at individual
        sub-question granularity rather than whole-topic, so a new topic that
        only overlaps 60% with a prior one still gets credit for the
        sub-questions it shares. Returns the matched finding_ids plus the
        real (not estimated) latency/tokens/cost recorded when those findings
        were originally produced -- see index_sub_question.
        """
        try:
            hits = self._subquestions_memory.search(sub_question, n_results=1)
        except Exception as e:
            logger.warning("Sub-question cache lookup failed: %s", e)
            return None

        if not hits:
            return None

        best = hits[0]
        if best.similarity < settings.KNOWLEDGE_REUSE_THRESHOLD:
            return None

        meta = best.metadata
        finding_ids = meta.get("finding_ids") or []
        if not finding_ids:
            return None
        return {
            "similarity": round(best.similarity, 4),
            "finding_ids": [int(i) for i in finding_ids],
            "latency_ms": meta.get("latency_ms"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cost_usd": meta.get("cost_usd"),
        }

    def index_sub_question(
        self,
        sub_question: str,
        finding_ids: List[int],
        latency_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        metadata: dict = {"finding_ids": list(finding_ids)}
        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms
        if prompt_tokens is not None:
            metadata["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            metadata["completion_tokens"] = completion_tokens
        if cost_usd is not None:
            metadata["cost_usd"] = cost_usd
        self._subquestions_memory.upsert(_make_id(sub_question), sub_question, metadata)

    def index_question(self, question_id: int, topic: str, conclusion_text: str) -> None:
        # Embed the topic alone (not the conclusion) so similarity reflects
        # "is this the same question", not diluted by unrelated answer text.
        self._questions_memory.upsert(
            f"question-{question_id}", topic, {"question_id": question_id, "topic": topic}
        )

    def index_finding(self, finding_id: int, question_id: int, text: str) -> None:
        self._knowledge_memory.upsert(
            f"finding-{finding_id}",
            text,
            {"type": "finding", "id": finding_id, "question_id": question_id},
        )

    def index_conclusion(self, conclusion_id: int, question_id: int, text: str) -> None:
        self._knowledge_memory.upsert(
            f"conclusion-{conclusion_id}",
            text,
            {"type": "conclusion", "id": conclusion_id, "question_id": question_id},
        )

    def search_knowledge(self, query: str, n_results: int = 8) -> list:
        hits = self._knowledge_memory.search(query, n_results=n_results)
        return [
            {
                "type": hit.metadata.get("type"),
                "id": hit.metadata.get("id"),
                "question_id": hit.metadata.get("question_id"),
                "text": hit.document,
                "similarity": round(hit.similarity, 4),
            }
            for hit in hits
        ]
