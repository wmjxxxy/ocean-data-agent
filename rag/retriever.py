"""Retrieve top-k knowledge chunks for a question."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from rag.embed_model import load_embedder

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "rag_db"
COLLECTION = "ocean_knowledge"


@lru_cache(maxsize=1)
def _model():
    return load_embedder()


@lru_cache(maxsize=1)
def _collection():
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"未找到向量库 {DB_DIR}。请先运行: python rag/ingest.py"
        )
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(COLLECTION)


def retrieve(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Return list of {id, text, source, distance}."""
    collection = _collection()
    query_embedding = _model().encode(
        [question],
        normalize_embeddings=True,
    )[0].tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result.get("distances", [[]])[0]
    out: list[dict[str, Any]] = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        out.append(
            {
                "id": f"S{i}",
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": float(dists[i - 1]) if dists else None,
            }
        )
    return out
