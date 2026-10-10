"""
Streamlit UI for the ocean research data agent.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent import run_agent

st.set_page_config(page_title="海洋科研数据分析智能体", layout="wide")

st.title("海洋科研数据分析智能体")
st.caption("自然语言 → Router → Tool Calling / RAG → 结论与图")

with st.sidebar:
    st.header("数据与说明")
    st.markdown(
        """
**可用数据集**
- EN4 / ORAS5 / CMIP6（MME）

**变量**：声速 `ssp`  
**深度**：5 / 100 / 500 / 1000 / 2000 m

**数值示例**
- 比较CMIP6和EN4在1000m的声速差异
- 画一下1000m Bias空间分布
- 比较100/500/1000/2000m，哪个RMSE最大

**知识示例（RAG）**
- 为什么声道轴搜索范围是100–3000m
- PCHIP为什么适合垂向插值
"""
    )
    st.divider()
    st.markdown("日志：`logs/agent_runs.jsonl`")

default_q = "为什么声道轴搜索范围设置为100–3000m？"
query = st.text_area("请输入分析问题", value=default_q, height=100)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("分析", type="primary", use_container_width=True)

if run:
    if not query.strip():
        st.warning("请先输入问题。")
    else:
        with st.spinner("智能体分析中…"):
            try:
                out = run_agent(query.strip())
            except Exception as e:  # noqa: BLE001
                st.error(f"运行失败：{e}")
                st.stop()

        st.caption(
            f"run_id: `{out.get('run_id', '')}` ｜ route: **{out.get('route', 'data')}**"
        )

        st.subheader("执行步骤")
        if not out.get("steps"):
            if out.get("route") in ("knowledge", "hybrid"):
                st.info("本次走知识检索（RAG），无数值工具调用或工具步骤见下。")
            else:
                st.info("本次未调用工具（模型直接回答）。")
        if out.get("steps"):
            for i, step in enumerate(out["steps"], 1):
                ok = step.get("status") == "SUCCESS" or (
                    "result" in step and not step.get("error")
                )
                label = f"{i}. {'✅' if ok else '❌'} {step.get('tool')}"
                with st.expander(label, expanded=True):
                    st.json(
                        {
                            "arguments": step.get("arguments"),
                            "status": step.get("status"),
                            "result": step.get("result"),
                            "error": step.get("error"),
                        }
                    )

        st.subheader("分析结果")
        st.markdown(out["answer"])

        if out.get("sources"):
            st.subheader("资料来源")
            for s in out["sources"]:
                st.markdown(f"- **[{s.get('id', '')}]** `{s.get('source', '')}`")

        if out.get("figures"):
            st.subheader("图片")
            for fig in out["figures"]:
                path = Path(fig)
                if path.is_file():
                    st.image(str(path), caption=path.name, use_container_width=True)
                else:
                    st.warning(f"找不到图片：{fig}")
