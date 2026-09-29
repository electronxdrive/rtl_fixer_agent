"""Small persistent Chroma store for reviewed RCA lessons."""

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class KnowledgeBase:
    def __init__(self, path: Path, read_only: bool = False):
        self.read_only = read_only
        if read_only and not path.exists():
            raise FileNotFoundError(f"Frozen KB does not exist: {path}")
        self.client = chromadb.PersistentClient(path=str(path))
        embedding = OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small")
        self.collection = (self.client.get_collection("rca_lessons", embedding_function=embedding)
                           if read_only else self.client.get_or_create_collection(
                               "rca_lessons", embedding_function=embedding,
                               metadata={"hnsw:space": "cosine"}))

    def hints(self, description: str, category: str, limit: int = 3) -> list[dict]:
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(
            query_texts=[description],
            n_results=min(limit, count),
            where={"category": category},
        )
        return [
            {"lesson": doc, "task_id": meta["task_id"]}
            for doc, meta in zip(
                result["documents"][0], result["metadatas"][0]
            )
        ]

    def save(self, task: dict, repair: dict) -> None:
        if self.read_only:
            raise RuntimeError("The evaluation KB is frozen")
        # Store the short, reviewable RCA reasoning trail and verification state.
        grade_result = repair.get("grade", {})
        verification = (f"simulator passed ({grade_result.get('samples', 0)} samples)"
                        if grade_result.get("passed") else "provisional training lesson")
        document = (f"Problem: {task['description'][:200]}\n"
                    f"Intended behavior: {repair['intended_behavior']}\n"
                    f"Parameters to check: {repair['parameters_to_check']}\n"
                    f"Look for: {repair['diagnostic_cue']}\n"
                    f"Root cause: {repair['root_cause']}\n"
                    f"Evidence: {repair['evidence']}\n"
                    f"Fix: {repair['fix_summary']}\n"
                    f"Verification: {verification}")
        self.collection.upsert(
            ids=[task["id"]],
            documents=[document],
            metadatas=[{"category": task["category"], "task_id": task["id"]}],
        )

    def count(self) -> int:
        return self.collection.count()

    def export(self) -> list[dict]:
        items = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "lesson": document, "metadata": metadata}
            for id_, document, metadata in zip(
                items["ids"], items["documents"], items["metadatas"]
            )
        ]

