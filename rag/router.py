"""Simple LLM router: data | knowledge | hybrid."""

from __future__ import annotations

import os
import re
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv((__import__("pathlib").Path(__file__).resolve().parents[1]) / ".env")

Route = Literal["data", "knowledge", "hybrid"]

ROUTER_PROMPT = """判断用户问题属于哪一种，只返回一个词：data 或 knowledge 或 hybrid。

data：需要读取 NetCDF、计算统计量、比较 Bias/RMSE、绘图、提取深度、查看文件变量/维度等数值分析。
knowledge：询问科研方法、定义、为什么这样设置、PCHIP/WOA102/声道轴原理、数据集说明、指标含义等文档知识。
hybrid：明确要求既要真实计算数值，又要结合方法文档解释（例如“计算1000m Bias并解释方法含义”）。

注意：
- “为什么1000m Bias最大/含义是什么”这类解释题 → knowledge
- “计算/比较/画/查看文件变量” → data
- 只有同时要求计算+解释方法时才用 hybrid

不要解释，只输出：data / knowledge / hybrid
"""


def route_question(question: str) -> Route:
    """Route with light rules first, then LLM fallback."""
    q = question.strip().lower()

    # Explicit hybrid: compute + explain method
    if re.search(r"(计算|比较).{0,30}(并解释|结合.{0,10}方法|说明如何解读)", q):
        return "hybrid"
    if "解释方法含义" in q or "并结合评价方法" in q:
        return "hybrid"

    # Conceptual / why questions → knowledge (even if Bias/RMSE/depth appear)
    if re.search(r"^(为什么|为何|什么是|啥是)", q) or re.search(
        r"(是什么|含义是什么|分别衡量|能不能|是否能|能否|是否意味着|怎么定的|怎么定义|如何定义|如何表述|适合做什么参考)",
        q,
    ):
        # unless clearly asking to run a plot/calc on files
        if not re.search(r"(计算|比较|画|绘制|查看|读取|inspect|统计|提取).{0,20}(文件|变量|图|rmse|bias|声速)", q):
            if re.search(
                r"(为什么|是什么|含义|衡量|定义|原理|方法|表述|pchip|声道轴|woa|mme|en4|oras5|bias|rmse|相关|偏差|插值|选取)",
                q,
            ):
                return "knowledge"

    # Midterm-style numeric facts asked as knowledge (approx numbers in docs)
    if re.search(r"(大约|约为).{0,20}(bias|rmse)", q) and re.search(r"(评价|口径|表述|中期)", q):
        return "knowledge"
    if "先算声速再平均" in q or "怎么定义" in q and "mme" in q:
        return "knowledge"

    # Hard: "EN4有哪些数据" = dataset description (knowledge); "有哪些变量" on file = data
    if re.search(r"(有哪些数据|资料适合|起什么作用|选取原则|一共选取)", q):
        return "knowledge"
    if re.search(r"(有哪些变量|哪些维度|数据集结构|做inspect|inspect一下)", q):
        return "data"

    # Explicit data/analysis requests
    if re.search(
        r"(分析|计算|提取|比较|画|绘制|统计|读取|查看).{0,20}(chlorophyll|叶绿素|thetao|声速|bias|rmse|分布|偏差图|文件|变量|维度|\d+\s*m)",
        q,
    ):
        return "data"
    if "chlorophyll" in q or "叶绿素" in q:
        return "data"
    if re.search(r"(用pchip|插到\d+\s*m|插值到)", q):
        return "data"
    if re.search(r"(差多少|是多少|min和max|一致性)", q) and re.search(
        r"(cmip6|en4|oras5|声速|bias|rmse|\d+\s*m)", q
    ):
        return "data"

    knowledge_kw = [
        "为什么",
        "是什么",
        "原理",
        "如何定义",
        "怎么识别",
        "pchip",
        "插值",
        "声道轴",
        "搜索范围",
        "100-3000",
        "100–3000",
        "方法",
        "含义",
        "woa102",
        "衡量什么",
        "敏感性",
        "teos",
    ]
    data_kw = [
        "比较",
        "bias",
        "rmse",
        "画",
        "绘图",
        "绘制",
        "平均",
        "统计",
        "分布",
        "差异",
        "相关",
        "1000m",
        "500m",
        "2000m",
        "分析",
        "计算",
        "查看",
        "读取",
        "inspect",
    ]

    has_k = any(k in q for k in knowledge_kw)
    has_d = any(k in q for k in data_kw)

    if re.search(r"(比较|计算|画|绘制).{0,20}(bias|rmse|声速|分布|偏差)", q):
        if re.search(r"(并解释|结合方法|说明如何解读)", q):
            return "hybrid"
        return "data"
    if has_k and not has_d:
        return "knowledge"
    if has_d and not has_k:
        return "data"
    if has_k and has_d:
        # prefer knowledge for pure "why bias" style if no action verb beyond keywords
        if re.search(r"为什么.{0,20}(bias|rmse|偏差)", q) and not re.search(
            r"(计算|比较|画|绘制)", q
        ):
            return "knowledge"
        return "hybrid"

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "knowledge" if has_k else "data"
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    if "hybrid" in text:
        return "hybrid"
    if "knowledge" in text:
        return "knowledge"
    return "data"
