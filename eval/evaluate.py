"""
Evaluation runner.

Usage:
  python eval/evaluate.py --route-only
  python eval/evaluate.py --full

Reports:
  - Router accuracy (all cases)
  - RAG Hit@4 (knowledge cases with expected_source)
  - Hard-case breakdown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cases() -> list[dict]:
    path = Path(__file__).resolve().parent / "eval_cases.jsonl"
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def eval_route_and_rag() -> None:
    from rag.retriever import retrieve
    from rag.router import route_question

    cases = load_cases()
    route_ok = 0
    rag_ok = 0
    rag_n = 0
    fail_ids: list[str] = []
    tag_stats: Counter[str] = Counter()
    tag_ok: Counter[str] = Counter()

    print(f"=== Route + RAG Hit@4  (N={len(cases)}) ===")
    for c in cases:
        q = c["question"]
        pred = route_question(q)
        expect = c.get("expected_route")
        ok = pred == expect
        route_ok += int(ok)
        tags = c.get("tags") or ["untagged"]
        for t in tags:
            tag_stats[t] += 1
            tag_ok[t] += int(ok)
        mark = "OK" if ok else "FAIL"
        if not ok:
            fail_ids.append(c["id"])
        print(f"[{c['id']}] route={pred} expected={expect} {mark}")

        if expect == "knowledge" and c.get("expected_source"):
            rag_n += 1
            hits = retrieve(q, top_k=4)
            sources = {h["source"] for h in hits}
            hit = c["expected_source"] in sources
            rag_ok += int(hit)
            print(f"       RAG Hit@4: {sources} -> {'OK' if hit else 'MISS'}")

    print(f"\nRoute accuracy: {route_ok}/{len(cases)} = {route_ok/len(cases):.1%}")
    if rag_n:
        print(f"RAG Hit@4: {rag_ok}/{rag_n} = {rag_ok/rag_n:.1%}")
    print("\nBy tag:")
    for t in sorted(tag_stats):
        print(f"  {t}: {tag_ok[t]}/{tag_stats[t]} = {tag_ok[t]/tag_stats[t]:.1%}")
    if fail_ids:
        print("Failed ids:", ", ".join(fail_ids))
    else:
        print("Failed ids: (none)")


def eval_full() -> None:
    from agent import run_agent

    cases = load_cases()
    tool_ok = 0
    tool_n = 0
    task_ok = 0
    t0 = time.perf_counter()

    print("=== Full agent evaluation (uses API) ===")
    for c in cases:
        q = c["question"]
        out = run_agent(q)
        tools = [s.get("tool") for s in out.get("steps", [])]
        expect_tool = c.get("expected_tool")
        if expect_tool:
            tool_n += 1
            ok = expect_tool in tools
            tool_ok += int(ok)
            print(f"[{c['id']}] tools={tools} expected={expect_tool} {'OK' if ok else 'FAIL'}")
        else:
            print(f"[{c['id']}] route={out.get('route')} tools={tools}")

        if out.get("answer"):
            if c.get("expected_error"):
                text = out["answer"] + json.dumps(out.get("steps", []), ensure_ascii=False)
                if any(k in text for k in ("越界", "不存在", "FAILED", "error")):
                    task_ok += 1
            else:
                task_ok += 1

    elapsed = time.perf_counter() - t0
    print(f"\nTask success (heuristic): {task_ok}/{len(cases)} = {task_ok/len(cases):.1%}")
    if tool_n:
        print(f"Tool selection (contains expected): {tool_ok}/{tool_n} = {tool_ok/tool_n:.1%}")
    print(f"Avg wall time / case: {elapsed/len(cases):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full:
        eval_full()
    else:
        eval_route_and_rag()


if __name__ == "__main__":
    main()
