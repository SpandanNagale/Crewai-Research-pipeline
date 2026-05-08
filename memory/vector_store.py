import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import hashlib
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


class ResearchMemory:
    """
    Persistent vector store for the Researcher agent.
    Stores past findings so repeated sub-questions don't trigger redundant searches.
    """

    def __init__(self, collection_name: str = "research_findings"):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
        )

    def _make_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def store(self, question: str, answer: str, source: str = "") -> None:
        """Store a question-answer pair in memory."""
        doc_id = self._make_id(question)
        self.collection.upsert(
            ids=[doc_id],
            documents=[answer],
            metadatas=[{"question": question, "source": source}],
        )

    def query(self, question: str, n_results: int = 2) -> List[Dict]:
        """
        Retrieve the most relevant past findings for a question.
        Returns a list of dicts with 'answer', 'question', 'source'.
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=min(n_results, self.collection.count() or 1),
        )
        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({
                "answer": doc,
                "question": meta.get("question", ""),
                "source": meta.get("source", ""),
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Wipe all stored findings (useful for fresh research runs)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.ef,
        )
