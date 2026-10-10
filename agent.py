"""
Ocean research data agent: LLM + Tool Calling loop.

Uses OpenAI-compatible API (DeepSeek / Qwen / OpenAI).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from logging_utils import log_event
from rag.answer import answer_with_rag
from rag.router import route_question
from tools import (
    calculate_statistics,
    compare_datasets,
    extract_depth,
    inspect_dataset,
    plot_bias_map,
    plot_map,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """你是海洋科研数据分析智能体。
你的职责是理解用户的自然语言分析需求，通过调用提供的 Python 工具完成 NetCDF 数据处理，再给出简洁、专业、审慎的中文结论。

规则：
1. 数值计算必须通过工具完成，不要自己估算 Bias/RMSE/均值等。
2. 可用数据集别名：EN4、ORAS5、CMIP6（CMIP6 指 MME 气候态）。
3. 主变量是声速 ssp，单位 m/s；样本深度为 5、100、500、1000、2000 m。
4. 用户说“比较差异/一致性”时，优先调用 compare_datasets；需要单场分布图时调用 plot_map；
   用户说“Bias空间分布/偏差分布/差值图”时，调用 plot_bias_map（默认 CMIP6 vs EN4，除非用户另有指定）。
4b. 若用户一次要求多个深度（如 100/500/1000/2000 m），请**连续多次**调用现有工具（每个深度一次），
   不要指望一次调用完成全部深度；全部结果返回后再汇总比较（例如哪个深度 RMSE 最大）。
5. 工具返回 error 时，不要假装成功；用中文明确说明原因（深度越界、变量不存在等），并给出可改深度/变量建议。
6. 最终回答包含：做了什么、关键数值、简要解读（1–3 句）。措辞要科研、稳妥，避免过度推断。

科学表述约束（非常重要）：
7. 声速同时受温度、盐度和压力/深度影响。在仅有 ssp Bias/RMSE、尚未比较温盐偏差时，
   不得断言“因为温度偏高/偏低”或“这是暖偏差/冷偏差”。
8. 对正/负 Bias，应表述为系统性偏差；可说明“可能与模式温盐结构模拟误差有关，
   具体贡献需结合温度、盐度偏差进一步分析”，不要把可能写成定论。
9. 空间相关系数请写“约为 x.xx，表明两者大尺度空间分布具有较高一致性”，
   不要用“高达”“非常一致”等夸张措辞，也不要把相关直接等同于物理机制已弄清。
10. 示例（1000 m、Bias 为正时）：“1000 m 处 CMIP6 声速整体高于 EN4，表明该深度存在较明显的
    系统性正偏差。该偏差可能与模式温盐结构模拟误差有关，具体贡献需要结合温度、盐度偏差进一步分析。”
"""

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": "查看数据集变量、维度、深度范围等元信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_or_path": {
                        "type": "string",
                        "description": "数据集别名：EN4 / ORAS5 / CMIP6，或文件路径",
                    }
                },
                "required": ["dataset_or_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_depth",
            "description": "提取指定深度层的场摘要（均值/范围/有效点数）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_or_path": {"type": "string"},
                    "variable": {"type": "string", "default": "ssp"},
                    "depth": {"type": "number", "description": "深度，单位 m"},
                },
                "required": ["dataset_or_path", "depth"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_statistics",
            "description": "计算指定数据集、变量、深度的 mean/std/min/max。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_or_path": {"type": "string"},
                    "variable": {"type": "string", "default": "ssp"},
                    "depth": {"type": "number"},
                },
                "required": ["dataset_or_path", "depth"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_datasets",
            "description": "比较两个数据集在指定深度的 Bias、RMSE、Correlation。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_dataset": {
                        "type": "string",
                        "description": "模式数据，如 CMIP6",
                    },
                    "reference_dataset": {
                        "type": "string",
                        "description": "参考数据，如 EN4 或 ORAS5",
                    },
                    "variable": {"type": "string", "default": "ssp"},
                    "depth": {"type": "number"},
                },
                "required": ["model_dataset", "reference_dataset", "depth"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_map",
            "description": "绘制单个数据集指定深度的空间分布图，返回 PNG 路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_or_path": {"type": "string"},
                    "variable": {"type": "string", "default": "ssp"},
                    "depth": {"type": "number"},
                    "title": {"type": "string"},
                },
                "required": ["dataset_or_path", "depth"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_bias_map",
            "description": (
                "绘制两个数据集在指定深度的 Bias（model−reference）空间分布图。"
                "用户要看 Bias/偏差/差值空间分布时使用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_dataset": {
                        "type": "string",
                        "description": "模式数据，默认常用 CMIP6",
                    },
                    "reference_dataset": {
                        "type": "string",
                        "description": "参考数据，默认常用 EN4",
                    },
                    "variable": {"type": "string", "default": "ssp"},
                    "depth": {"type": "number"},
                    "title": {"type": "string"},
                },
                "required": ["model_dataset", "reference_dataset", "depth"],
            },
        },
    },
]

TOOL_IMPL: dict[str, Callable[..., Any]] = {
    "inspect_dataset": inspect_dataset,
    "extract_depth": extract_depth,
    "calculate_statistics": calculate_statistics,
    "compare_datasets": compare_datasets,
    "plot_map": plot_map,
    "plot_bias_map": plot_bias_map,
}

PLOT_TOOLS = {"plot_map", "plot_bias_map"}


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 API Key。请在项目根目录创建 .env，写入：\n"
            "OPENAI_API_KEY=sk-你的密钥\n"
            "OPENAI_BASE_URL=https://api.deepseek.com\n"
            "OPENAI_MODEL=deepseek-chat"
        )
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def _run_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    run_id: str,
    user_query: str,
) -> dict[str, Any]:
    if name not in TOOL_IMPL:
        payload = {"error": f"未知工具: {name}"}
        log_event(
            {
                "type": "tool_call",
                "run_id": run_id,
                "user_query": user_query,
                "tool": name,
                "arguments": arguments,
                "status": "FAILED",
                "elapsed_s": 0.0,
                "result": None,
                "error": payload["error"],
            }
        )
        return payload

    t0 = time.perf_counter()
    try:
        result = TOOL_IMPL[name](**arguments)
        status = "SUCCESS"
        payload: dict[str, Any] = {"result": result}
    except Exception as e:  # noqa: BLE001 - surface tool errors to the LLM
        status = "FAILED"
        payload = {"error": str(e)}
        result = None
    elapsed = time.perf_counter() - t0
    log_event(
        {
            "type": "tool_call",
            "run_id": run_id,
            "user_query": user_query,
            "tool": name,
            "arguments": arguments,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "result": result if status == "SUCCESS" else None,
            "error": payload.get("error"),
        }
    )
    return payload


def _collect_figure(tool_name: str, tool_out: dict[str, Any], figures: list[str]) -> None:
    if tool_name not in PLOT_TOOLS:
        return
    result = tool_out.get("result")
    if isinstance(result, dict):
        fig = result.get("figure_path")
        if fig:
            figures.append(fig)


def run_tool_agent(user_query: str, max_rounds: int = 8, run_id: str | None = None) -> dict[str, Any]:
    """
    Tool-calling agent loop (data path):
    user -> LLM -> (optional tool calls) -> tool results -> LLM -> final answer
    """
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    run_id = run_id or uuid.uuid4().hex[:12]
    t_run = time.perf_counter()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    log_event(
        {
            "type": "user_query",
            "run_id": run_id,
            "user_query": user_query,
            "route": "data",
            "status": "STARTED",
        }
    )

    steps: list[dict[str, Any]] = []
    figures: list[str] = []

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SPECS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_out = _run_tool(
                    name,
                    args,
                    run_id=run_id,
                    user_query=user_query,
                )
                step = {
                    "tool": name,
                    "arguments": args,
                    "status": "SUCCESS" if "result" in tool_out and "error" not in tool_out else "FAILED",
                    **tool_out,
                }
                steps.append(step)
                _collect_figure(name, tool_out, figures)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_out, ensure_ascii=False),
                    }
                )
            continue

        answer = (msg.content or "").strip()
        log_event(
            {
                "type": "final_answer",
                "run_id": run_id,
                "user_query": user_query,
                "route": "data",
                "status": "SUCCESS",
                "elapsed_s": round(time.perf_counter() - t_run, 3),
                "steps": len(steps),
                "tool_names": [s["tool"] for s in steps],
                "content": answer,
            }
        )
        return {
            "answer": answer,
            "steps": steps,
            "figures": figures,
            "sources": [],
            "route": "data",
            "run_id": run_id,
        }

    fallback = "达到最大工具调用轮次，请简化问题后重试。"
    log_event(
        {
            "type": "final_answer",
            "run_id": run_id,
            "user_query": user_query,
            "route": "data",
            "status": "FAILED",
            "elapsed_s": round(time.perf_counter() - t_run, 3),
            "steps": len(steps),
            "tool_names": [s["tool"] for s in steps],
            "content": fallback,
            "error": "max_rounds_exceeded",
        }
    )
    return {
        "answer": fallback,
        "steps": steps,
        "figures": figures,
        "sources": [],
        "route": "data",
        "run_id": run_id,
    }


def _merge_hybrid(user_query: str, data_out: dict[str, Any], rag_out: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Combine tool results and RAG evidence into one cautious answer."""
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    evidence = "\n".join(
        f"[{s['id']}] {s['source']}" for s in rag_out.get("sources", [])
    )
    prompt = f"""用户问题：{user_query}

数值分析结果：
{data_out.get('answer', '')}

知识资料回答：
{rag_out.get('answer', '')}

引用资料：
{evidence}

请合并为一份审慎中文回答：数值以工具结果为准；方法解释仅使用资料并保留 [S1] 等引用；
不得仅凭声速 Bias 断言温度机制。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你合并数值结果与资料引用，表述审慎。"},
            {"role": "user", "content": prompt},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()
    return {
        "answer": answer,
        "steps": data_out.get("steps", []),
        "figures": data_out.get("figures", []),
        "sources": rag_out.get("sources", []),
        "route": "hybrid",
        "run_id": run_id,
    }


def run_agent(user_query: str, max_rounds: int = 8) -> dict[str, Any]:
    """Public entry: route to Tool Calling and/or RAG."""
    run_id = uuid.uuid4().hex[:12]
    route = route_question(user_query)
    log_event(
        {
            "type": "route",
            "run_id": run_id,
            "user_query": user_query,
            "route": route,
            "status": "SUCCESS",
        }
    )

    if route == "knowledge":
        out = answer_with_rag(user_query, run_id=run_id)
        out["run_id"] = run_id
        return out

    if route == "hybrid":
        data_out = run_tool_agent(user_query, max_rounds=max_rounds, run_id=run_id)
        rag_out = answer_with_rag(user_query, run_id=run_id)
        return _merge_hybrid(user_query, data_out, rag_out, run_id)

    return run_tool_agent(user_query, max_rounds=max_rounds, run_id=run_id)


def main() -> None:
    import sys

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "比较CMIP6和EN4在1000m的声速差异，并给出Bias和RMSE。"
        print(f"（未传入问题，使用默认示例）\n{query}\n")

    out = run_agent(query)
    print(f"\nrun_id: {out.get('run_id')}")
    print("\n=== 执行步骤 ===")
    for i, step in enumerate(out["steps"], 1):
        status = step.get("status", "OK" if "result" in step else "ERR")
        print(f"{i}. [{status}] {step['tool']} args={step.get('arguments')}")
        if step.get("error"):
            print(f"   error: {step['error']}")
        elif "result" in step:
            print(f"   result: {json.dumps(step['result'], ensure_ascii=False)}")

    if out["figures"]:
        print("\n=== 图片 ===")
        for fig in out["figures"]:
            print(fig)

    print("\n=== 分析结果 ===")
    print(out["answer"])


if __name__ == "__main__":
    main()
