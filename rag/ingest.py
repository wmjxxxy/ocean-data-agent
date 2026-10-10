"""Build local vector knowledge base from markdown docs."""

from __future__ import annotations

from pathlib import Path

import chromadb

from rag.embed_model import load_embedder

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
DB_DIR = ROOT / "rag_db"
COLLECTION = "ocean_knowledge"


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += max(chunk_size - overlap, 1)
    return chunks


def main() -> None:
    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"未找到知识文档: {KNOWLEDGE_DIR}")

    model = load_embedder()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001
        pass
    collection = client.get_or_create_collection(name=COLLECTION)

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    counter = 0

    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "source": file.name,
                    "chunk_index": i,
                    "title": file.stem,
                }
            )
            ids.append(f"{file.stem}_{counter}")
            counter += 1

    print(f"Encoding {len(documents)} chunks...")
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"已建立知识库：{len(documents)} chunks → {DB_DIR}")
    for f in files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
