"""
Knowledge reuse: before running a full pipeline for a new question, embed
the topic and compare it against past research_questions via ChromaDB. If a
close-enough match exists, surface its existing findings/conclusions instead
of (or in addition to) re-running full research.

Also maintains a second, finer-grained index over individual findings and
conclusions for the GET /knowledge/search semantic-search endpoint.
"""
import logging
from typing import Optional

from config.settings import settings
from db import repository
from memory.vector_store import ResearchMemory

logger = logging.getLogger(__name__)

QUESTIONS_COLLECTION = "research_questions_index"
KNOWLEDGE_COLLECTION = "knowledge_index"


class KnowledgeReuseChecker:
    def __init__(self):
        self._questions_memory = ResearchMemory(collection_name=QUESTIONS_COLLECTION)
        self._knowledge_memory = ResearchMemory(collection_name=KNOWLEDGE_COLLECTION)

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
