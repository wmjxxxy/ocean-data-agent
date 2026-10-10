"""Shared embedding model path for offline-first RAG."""

from __future__ import annotations

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
HF_MODEL_ID = "BAAI/bge-small-zh-v1.5"


def resolve_embed_model_path() -> str:
    """Prefer project-local model; fall back to HF id."""
    required = [
        LOCAL_MODEL_DIR / "config.json",
        LOCAL_MODEL_DIR / "modules.json",
        LOCAL_MODEL_DIR / "tokenizer.json",
    ]
    has_weight = (LOCAL_MODEL_DIR / "model.safetensors").is_file() or (
        LOCAL_MODEL_DIR / "pytorch_model.bin"
    ).is_file()
    if LOCAL_MODEL_DIR.is_dir() and has_weight and all(p.is_file() for p in required):
        return str(LOCAL_MODEL_DIR)
    return HF_MODEL_ID


def load_embedder() -> SentenceTransformer:
    """
    Load embedder with a complete offline-first path:
    1) project local folder models/bge-small-zh-v1.5
    2) HF cache / mirror download
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    path = resolve_embed_model_path()
    print(f"Embedding model: {path}")
    if path == HF_MODEL_ID:
        try:
            return SentenceTransformer(path, local_files_only=True)
        except Exception:
            print("本地缓存未命中，尝试通过镜像下载…")
            return SentenceTransformer(path)
    # local folder: never need network
    return SentenceTransformer(path)
