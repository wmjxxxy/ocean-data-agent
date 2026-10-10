"""RAG answering with citations."""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from logging_utils import log_event
from rag.retriever import retrieve

ROOT = __file__
load_dotenv((__import__("pathlib").Path(__file__).resolve().parents[1]) / ".env")

RAG_SYSTEM = """你是海洋科研知识助手。
请仅根据提供的资料回答问题。
如果资料不足，请明确说“现有资料不足以支持该结论”，不得自行补充未经资料支持的科研结论。
回答使用简洁中文，并在相关句末使用 [S1]、[S2] 等标记引用来源。
不要编造温盐归因或论文中没有的数值。
"""


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 API Key，请配置 .env")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )


def answer_with_rag(
    question: str,
    *,
    run_id: str | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    hits = retrieve(question, top_k=top_k)
    evidence = "\n\n".join(
        f"[{h['id']}]\nsource: {h['source']}\n{h['text']}" for h in hits
    )
    user_prompt = (
        f"用户问题：\n{question}\n\n资料：\n{evidence}\n\n"
        "请基于资料作答，并标注 [S1]/[S2] 等引用。"
    )

    client = _client()
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()
    elapsed = round(time.perf_counter() - t0, 3)
    sources = [{"id": h["id"], "source": h["source"]} for h in hits]

    if run_id:
        log_event(
            {
                "type": "rag_answer",
                "run_id": run_id,
                "user_query": question,
                "route": "knowledge",
                "status": "SUCCESS",
                "elapsed_s": elapsed,
                "retrieved_sources": [h["source"] for h in hits],
                "result": {"answer": answer, "sources": sources},
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "hits": hits,
        "elapsed_s": elapsed,
        "route": "knowledge",
        "steps": [],
        "figures": [],
    }
