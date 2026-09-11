"""
Knowledge reuse: before running a full pipeline for a new question, embed
the topic and compare it against past research_questions via ChromaDB. If a
close-enough match exists, surface its existing findings/conclusions instead
of (or in addition to) re-running full research.

Also maintains a second, finer-grained index over individual findings and
conclusions for the GET /knowledge/search semantic-search endpoint.
"""
import logging
from typing import List, Optional

from config.settings import settings
from db import repository
from memory.vector_store import ResearchMemory

logger = logging.getLogger(__name__)

QUESTIONS_COLLECTION = "research_questions_index"
KNOWLEDGE_COLLECTION = "knowledge_index"
SUBQUESTIONS_COLLECTION = "sub_questions_index"


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
        if self._questions_memory.count() == 0:
            return None

        try:
            results = self._questions_memory.collection.query(
                query_texts=[topic],
                n_results=1,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("Knowledge reuse similarity search failed: %s", e)
            return None

        ids = results.get("ids", [[]])[0]
        if not ids:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance  # cosine space: distance = 1 - cosine_similarity
        if similarity < settings.KNOWLEDGE_REUSE_THRESHOLD:
            return None

        question_id = int(results["metadatas"][0][0]["question_id"])
        trace = repository.get_trace(question_id)
        if trace is not None:
            trace["similarity"] = round(similarity, 4)
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
        if self._subquestions_memory.count() == 0:
            return None
        try:
            results = self._subquestions_memory.collection.query(
                query_texts=[sub_question],
                n_results=1,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("Sub-question cache lookup failed: %s", e)
            return None

        ids = results.get("ids", [[]])[0]
        if not ids:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance
        if similarity < settings.KNOWLEDGE_REUSE_THRESHOLD:
            return None

        meta = results["metadatas"][0][0]
        finding_ids = [int(x) for x in meta["finding_ids"].split(",") if x]
        if not finding_ids:
            return None
        return {
            "similarity": round(similarity, 4),
            "finding_ids": finding_ids,
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
        # Chroma metadata values must be scalar (str/int/float/bool) -- finding_ids
        # is stored as a comma-joined string and parsed back in find_similar_subquestion.
        metadata: dict = {"finding_ids": ",".join(str(i) for i in finding_ids)}
        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms
        if prompt_tokens is not None:
            metadata["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            metadata["completion_tokens"] = completion_tokens
        if cost_usd is not None:
            metadata["cost_usd"] = cost_usd
        self._subquestions_memory.collection.upsert(
            ids=[self._subquestions_memory._make_id(sub_question)],
            documents=[sub_question],
            metadatas=[metadata],
        )

    def index_question(self, question_id: int, topic: str, conclusion_text: str) -> None:
        # Embed the topic alone (not the conclusion) so similarity reflects
        # "is this the same question", not diluted by unrelated answer text.
        self._questions_memory.collection.upsert(
            ids=[f"question-{question_id}"],
            documents=[topic],
            metadatas=[{"question_id": question_id, "topic": topic}],
        )

    def index_finding(self, finding_id: int, question_id: int, text: str) -> None:
        self._knowledge_memory.collection.upsert(
            ids=[f"finding-{finding_id}"],
            documents=[text],
            metadatas=[{"type": "finding", "id": finding_id, "question_id": question_id}],
        )

    def index_conclusion(self, conclusion_id: int, question_id: int, text: str) -> None:
        self._knowledge_memory.collection.upsert(
            ids=[f"conclusion-{conclusion_id}"],
            documents=[text],
            metadatas=[{"type": "conclusion", "id": conclusion_id, "question_id": question_id}],
        )

    def search_knowledge(self, query: str, n_results: int = 8) -> list:
        if self._knowledge_memory.count() == 0:
            return []
        results = self._knowledge_memory.collection.query(
            query_texts=[query],
            n_results=min(n_results, self._knowledge_memory.count()),
            include=["metadatas", "documents", "distances"],
        )
        output = []
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            output.append(
                {
                    "type": meta.get("type"),
                    "id": meta.get("id"),
                    "question_id": meta.get("question_id"),
                    "text": doc,
                    "similarity": round(1 - distance, 4),
                }
            )
        return output
