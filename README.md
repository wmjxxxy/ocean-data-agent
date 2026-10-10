# 海洋科研数据分析智能体

自然语言驱动的海洋 NetCDF 分析 Agent（MVP / POC）。

**流程：** 用户提问 → LLM 理解意图 → Tool Calling 选择工具 → Python/xarray 计算 → 日志记录 → 中文结论（可附图）

## 功能范围（领域智能体）

面向海洋声速气候态数据，支持：

- 数据集检查
- 指定深度提取与统计
- CMIP6 / EN4 / ORAS5 对比（Bias / RMSE / 空间相关）
- 单场空间分布与 Bias 空间分布绘图
- 深度/变量异常友好提示
- 执行日志追溯（用户问题、工具、参数、结果、耗时、状态）
- LLM 结论科学表述约束（避免仅凭声速 Bias 断言温盐机制）

样本深度：`5 / 100 / 500 / 1000 / 2000 m`  
主变量：`ssp`（声速，m/s）

## 项目结构

```text
ocean-data-agent/
├── app.py                 # Streamlit 界面
├── agent.py               # LLM + Tool Calling 循环
├── logging_utils.py       # 可观测性日志
├── tools/                 # 确定性 Python 工具
│   ├── dataset_tools.py
│   ├── statistics_tools.py
│   ├── comparison_tools.py
│   └── plot_tools.py
├── data/                  # 小型样本 NetCDF（可随仓库分发）
├── outputs/figures/       # 生成图片
├── logs/                  # agent_runs.jsonl
├── .env.example           # API 配置模板（勿提交真实 Key）
├── test_tools.py          # 工具层冒烟测试
└── requirements.txt
```

## 环境准备

建议使用已有科研环境（含 xarray / matplotlib），并安装：

```powershell
pip install -r requirements.txt
```

### 配置 API（DeepSeek 示例）

```powershell
copy .env.example .env
notepad .env
```

`.env` 内容：

```env
OPENAI_API_KEY=sk-你的密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

> 兼容任意 OpenAI 协议接口（通义等只需改 Base URL 与模型名）。  
> **不要把 `.env` 提交到 Git。**

## 快速开始

### 1）工具层自测（不需要 API）

```powershell
python test_tools.py
```

### 2）命令行智能体

```powershell
python agent.py 比较CMIP6和EN4在1000m的声速差异，并给出Bias和RMSE
```

### 3）网页界面

```powershell
streamlit run app.py
```

浏览器打开后输入问题，点击「分析」。

示例：`画一下1000m Bias空间分布` → 调用 `plot_bias_map` → Streamlit 展示图片。

### 4）知识库（RAG，基于中期报告）

知识文档在 `rag/knowledge/`（已从《硕士中期报告-王梦佳》提炼）：
- `sound_channel.md` 声道轴 100–3000 m、三点二次
- `interpolation.md` PCHIP / WOA102 / TEOS-10
- `datasets.md` CMIP6 / EN4 / ORAS5 / MME
- `cmip6_method.md` Bias/RMSE 与中期评价结论

首次或更新文档后需重建向量库：

```powershell
pip install chromadb sentence-transformers
python rag/ingest.py
```

然后可问：`为什么声道轴搜索范围设置为100–3000m？`

## 工具一览

| 工具 | 作用 |
|------|------|
| `inspect_dataset` | 查看变量/维度/深度 |
| `extract_depth` | 提取指定深度摘要 |
| `calculate_statistics` | mean/std/min/max |
| `compare_datasets` | Bias / RMSE / Correlation |
| `plot_map` | 单场空间分布图 |
| `plot_bias_map` | Bias 空间分布图 |

## 典型验证结果（POC）

问题：比较 CMIP6 与 EN4 在 1000 m 的声速差异  

| 指标 | 数值 |
|------|------|
| Bias | ≈ +4.83 m/s |
| RMSE | ≈ 5.92 m/s |
| Correlation | ≈ 0.91 |

说明：Bias/RMSE 为面积加权（`cos(lat)`），与科研评估脚本逻辑一致。

界面截图见 `docs/screenshots/`。

## 设计原则（面试可讲）

1. **LLM 负责推理，Python 负责计算**：不让大模型直接算 RMSE。  
2. **Tool Calling**：模型声明要调用的函数与参数，本地执行后把结果回传。  
3. **Observability**：`logs/agent_runs.jsonl` 记录用户问题、工具名、参数、结果、耗时、成功/失败，并用 `run_id` 串起一次完整会话。  
4. **异常处理**：变量不存在、深度越界等返回可读错误，由 Agent 用中文说明，而不是直接崩溃。  
5. **Domain-specific**：只做海洋 NetCDF 声速分析，不做“万能科研 Agent”。

## 后续可扩展（未做则不要写进简历）

- Planner：多深度/多步骤任务拆解  
- Reviewer：结果合理性检查与重试  
- Docker 一键运行  
- FastAPI 服务化  

## 简历可用描述

见 [`docs/resume_snippet.md`](docs/resume_snippet.md)。
