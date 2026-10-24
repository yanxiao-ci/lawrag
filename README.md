# lawrag

中文法律领域的 RAG 问答系统。能精确命中法条或库内问答的问题走 MySQL / Redis / BM25 快通道，其余问题走「法条 + 问答双召回 → BGE 精排 → 大模型生成」的 RAG 通道。回答要求标注《法律名》第 X 条，生成后再做一次引用校验。

系统适合「试用期最长多久」「租房押金不退可以引哪条」这类法条依据问题，不提供个案结论，也不能替代律师。

目前全量检索 Recall@5 0.901、MRR 0.7919；RAGAS 忠实度 0.899、context_precision 0.713、context_recall 0.730。HR 合规检查另有一套 140 条 JD 评测，规则轨 P/R/F1 1.0、引用准确率 100%、干净 JD 0 误报。评测脚本都在 `tools/` 下，数字可以复算。

项目背景、架构和完整指标见 [`docs/项目介绍.md`](docs/项目介绍.md)。本服务作为独立的法规知识层，通过 HTTP / MCP 给上层 Agent 提供检索、取条和合规检查；两个仓库不共享代码、数据库和索引。

## 1. 覆盖范围

覆盖全国人大及其常委会制定的法律原文（宪法、基本法律、法律、条例和办法等），共 288 部现行有效法律。

不覆盖：

- 行政法规、司法解释、部门规章、地方性法规、自治条例和国际条约；
- 裁判文书、判例、典型案例、合同模板和法律意见书；
- 任何形式的个案结论。

因此它适合回答「《劳动合同法》第 X 条怎么规定」这类问题；对「某地最新政策是什么」「我这个案子能不能赢」，只能给出法律层面的通用依据。

## 2. 数据规模

| 数据 | 位置 | 规模 |
|---|---|---:|
| 法律原文 | `rag_qa/data/law_data/` | 292 个文件，4.2 MB（288 部现行 + 4 个历史版本） |
| 法条表 | MySQL `law_articles` | 21,565 条（现行有效 21,312，历史版本 253） |
| 法条向量 | Milvus `lawrag` | 21,565 个子块（子块为单条法条，父块为同法连续 5 条） |
| 法律问答 | MySQL `law_faq` | 20,000 条 |
| 问答向量 | Milvus `lawfaq` | 20,000 条 |

数据来源：

- 法条主料：`KuugoRen/Chinese_Law`（ModelScope，97 部法律 txt），用 `tools/expand_law_corpus.py` 从本地全量法规库扩容到 292 部；
- 法律问答：`Brench/chinese_law_data_rag_ft`（HuggingFace 镜像，5 万条测试集），用 `tools/sample_law_faq.py` 按领域均衡、答案引用的法条必须真实存在，抽取 2 万条；
- 检索评测集：`ByronLeeee/CN-Law-Query-Retrieval-Dataset`（anchor / positive / negative）；
- 编章节层级：`Dusker/chinese-laws-pretrain`，用于补 `chapter` 元数据，可选。

采集脚本是 `python tools/collect_law_data.py`。HuggingFace 直连不通时默认走 `hf-mirror.com`，可以用 `HF_ENDPOINT` 覆盖。

## 3. 系统架构

完整流程图见 `docs/开发指南/RAG项目流程图说明.md`，源文件是 `docs/make_flowchart.py`，输出到 `docs/figures/`。

### 离线准备

1. 把 `rag_qa/data/law_data/*.txt` 按法条边界切分。子块是一条完整法条，父块是同一部法律连续 5 条，父子关系写入元数据，避免通用递归切分把法条截断。
2. 用 BGE-M3 生成稠密和稀疏向量，写入 Milvus 集合 `lawrag`。
3. 法条结构化信息写入 MySQL `law_articles`，包括编章节、公布日、施行日和时效状态。
4. 2 万条法律问答写入 MySQL `law_faq`，问题列表写入 Redis，供 BM25 检索使用。
5. 2 万条问答向量写入 Milvus 集合 `lawfaq`，作为 RAG 的第三召回源。

### 在线问答

```
用户提问
  ├─ 问候语正则 → 模板回复，不检索
  ├─ 法条号精确路由（“第 X 条” + 法律名）→ 直接查 law_articles
  ├─ 快问快答通道：Redis → 分词 → BM25 → 词面门控 → 命中即回标准答案
  └─ RAG 通道：意图识别 → 策略选择（直接 / HyDE / 子查询 / 回溯）
                → 法条 + 问答双路召回 → BGE-Reranker 精排
                → 拼接提示词 → 大模型流式生成 → 引用校验
```

快通道使用词面判据（Jaccard 和查询词覆盖率），不是分数阈值，避免把「相似但不同」的问题直接抢答。

### 评测

- `tools/eval_recall.py`：Recall@k、MRR，支持消融；
- `tools/eval_context_coverage.py`：上下文覆盖率，不依赖 LLM；
- `tools/tune_fast_channel_gate.py`：快通道门控标定；
- `rag_qa/rag_assessment/ragas_evaluate.py`：RAGAS 四项指标。

## 4. 快速开始

### 4.1 依赖服务

| 服务 | 默认地址 | 说明 |
|---|---|---|
| MySQL 8 | `192.168.229.128:3307` | 库 `subjects_kg`，表 `law_articles` / `law_faq` / 会话历史 |
| Redis | `192.168.229.128:6379` | 快通道缓存和会话缓存 |
| Milvus | `192.168.229.128:19530` | 库 `itcast`，集合 `lawrag` / `lawfaq` |

`docker-compose.yml` 只包含应用本身（`lawrag-app`），三个依赖服务需要自行部署。Milvus standalone 还需要 etcd 和 MinIO。

### 4.2 Python 环境

```bash
conda create -n lawrag python=3.10 -y
conda activate lawrag
pip install -r requirements.txt
```

安装 `torch` 时按显卡 CUDA 版本选官方 wheel。有 GPU 时，向量化和精排会自动使用（`vector_store.py` 里做了设备探测）。

### 4.3 模型

权重体积过大，已在 `.gitignore` 中排除。按下面的目录名放到 `rag_qa/models/`：

| 目录 | 用途 | 获取方式 |
|---|---|---|
| `bert-base-chinese` | 意图分类器底座 | HuggingFace / ModelScope |
| `bge-m3` | 稠密 + 稀疏向量 | ModelScope `BAAI/bge-m3` |
| `bge-reranker-large` | 精排（CrossEncoder） | ModelScope `BAAI/bge-reranker-large` |
| `bert_query_classifier_law` | 意图分类 | `python tools/train_law_classifier.py` 训练生成 |

### 4.4 配置

配置优先级：`.env` > `config.ini` > 代码内默认值。`.env` 已在 git 之外，本地新建即可：

```ini
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=subjects_kg

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your-password

MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_DATABASE_NAME=itcast
MILVUS_COLLECTION_NAME=lawrag

# 大模型：需要自己申请 DashScope（百炼）API Key
LLM_MODEL=qwen-plus
DASHSCOPE_API_KEY=sk-your-key-here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RETRIEVAL_K=5
CANDIDATE_M=2

VALID_SOURCES=["law"]
CUSTOMER_SERVICE_PHONE=13000000000
```

### 4.5 建库

```bash
# 1) 法条和问答写入 MySQL（会 TRUNCATE 后重写 law_articles / law_faq）
python tools/import_law_data.py

# 2) 法条向量库；语料换过必须加 --reset，否则新旧块会混在同一个集合
python tools/build_law_index.py --reset

# 3) 问答向量库
python tools/build_lawfaq_index.py --reset
```

### 4.6 启动

```bash
python app.py          # 或 uvicorn app:app --host 0.0.0.0 --port 8080
```

浏览器打开 <http://127.0.0.1:8080/>。注意 `0.0.0.0` 是监听地址，不是访问地址。

## 5. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/query` | 一次性问答 |
| WS | `/api/stream` | 流式问答（前端默认） |
| POST | `/api/create_session` | 新建会话 |
| GET / DELETE | `/api/history/{session_id}` | 查询 / 清空会话历史 |
| GET | `/health` | 存活检查 |
| GET | `/api/sources` | 可用知识库来源，当前为 `["law"]` |
| GET | `/api/domains` | 可用领域视图，当前为 `labor` |
| POST | `/api/search_law` | 检索-only，给 MCP 或 Agent 返回条文证据 |
| POST | `/api/get_article` | 按法律名 + 条号取全文 |
| POST | `/api/check_hr_compliance` | HR 合规检查：规则命中 + 闭集法条依据 + 风险等级 |
| GET | `/api/corpus_info` | 语料规模、领域和覆盖边界 |

`/api/check_hr_compliance` 的请求和响应：

```json
// 请求
{"text": "招聘后端工程师，限男性，35岁以下，试用期9个月。", "scope": "jd", "domain": "labor"}

// 响应
{
  "risk_level": "high",
  "findings": [{
    "category": "就业歧视",
    "span": "限男性",
    "severity": "high",
    "suggestion": "删除性别限制，改为与岗位相关的任职资格",
    "basis": [{"law_name": "中华人民共和国就业促进法", "article_no": "第二十七条", "quote": "……"}],
    "needs_human": false
  }],
  "disclaimer": "仅覆盖全国人大及其常委会制定的法律级文本，不含行政法规、司法解释、地方规定；本结果仅作合规提示，不构成法律意见。",
  "citation_verified": true,
  "llm_used": true
}
```

几个约束：

- 规则只负责定位片段、类别和建议，不产生条号。条号只来自 `law_hint` 的 MySQL 回查，或者严格劳动法领域的检索兜底。
- `quote` 去掉空白、统一全半角标点后，必须是 MySQL `article_text` 的连续子串，否则丢弃该 basis，并标记 `needs_human=true`。
- `citation_verified` 表示每个风险项都至少有一条通过逐字回查的 basis。干净文本没有风险项时为 true；有风险项但 basis 为空时为 false，同时标记 `needs_human=true`。
- LLM 只做闭集选条，超时 5 秒，失败回退到确定性 hint，不修改 span、category、severity。
- `risk_level` 由规则严重程度聚合，不交给 LLM。
- 当前只覆盖法律级文本，不含《女职工劳动保护特别规定》等行政法规。

## 6. MCP 工具

`mcp_server/server.py` 是薄封装，只调上面的 HTTP 接口，不 import `app.py`，不重复加载模型。

```bash
python mcp_server/server.py                  # stdio，开发或 Claude Code
python mcp_server/server.py streamable-http  # HTTP，部署，默认 127.0.0.1:8000/mcp
```

MCP server 只需要装有 `mcp` 的解释器（本机用 `edu_agent` 环境），不需要 lawrag 的 GPU 环境。想在 Claude Code 里直接使用，把 `mcp_server/example.mcp.json` 的 `lawrag` 段合并到项目的 `.mcp.json`。

工具：`search_law`、`get_article`、`check_hr_compliance`、`get_corpus_info`。

冒烟测试（需要 lawrag 已启动，解释器装有 `mcp`）：

```bash
python tools/mcp_smoke_test.py
```

Python 调用示例：

```python
import requests

r = requests.post(
    "http://127.0.0.1:8080/api/search_law",
    json={"query": "试用期最长多久", "top_k": 3, "domain": "labor"},
)
print(r.json()["results"])
```

## 7. 目录结构

```
lawrag/
├─ app.py                     # FastAPI 入口（HTTP + WebSocket 流式）
├─ new_main.py                # 在线问答总调度：问候语 → 条文号路由 → 快通道 → RAG
├─ config.ini / .env          # 配置（.env 覆盖 config.ini，已 gitignore）
├─ base/                      # 配置、日志、法律语料解析器
├─ mysql_qa/                  # 快问快答：MySQL / Redis / BM25 与门控
├─ rag_qa/
│  ├─ core/                   # 意图识别、策略选择、向量库、检索、提示词、引用校验
│  ├─ document_loaders/       # PDF / Word / PPT / 图片加载器
│  ├─ text_splitters/         # 法条边界切分器和通用中文切分
│  ├─ models/                 # 模型权重（不入库）
│  ├─ data/                   # 法律原文、分类数据、评测产物
│  └─ rag_assessment/         # RAGAS 评测脚本
├─ static/index.html          # 前端页面
├─ mcp_server/server.py       # MCP 薄封装
├─ tools/                     # 数据采集、建库、评测、调参脚本
└─ docs/                      # 流程图、项目概览与迭代记录
```

早期文件（旧入口、旧前端、旧语料、旧评测集）已移到 `_archive/` 并在 `.gitignore` 中排除，仍保留在本机和 git 历史里。

## 8. 评测

```bash
# 检索质量：Recall@k / MRR，含 dense / hybrid / +rerank 消融
python tools/eval_recall.py

# 上下文覆盖率，不依赖 LLM
python tools/eval_context_coverage.py

# 快通道门控标定
python tools/tune_fast_channel_gate.py

# RAGAS：建评测集 → 跑 RAG 通道收答案 → 判分
python tools/build_law_ragas_set.py
python tools/collect_law_ragas_answers.py --workers 8
python rag_qa/rag_assessment/ragas_evaluate.py \
  --data rag_qa/data/law_eval/ragas/law_ragas_answers.json \
  --judge-model qwen-max \
  --out rag_qa/data/law_eval/ragas/ragas_law.csv

# HR 合规：先启动 app.py
python tools/build_hr_compliance_eval.py          # 生成并逐条核对 140 条评测集
python tools/eval_hr_compliance.py --split dev    # 调规则和 Prompt 只用 dev
python tools/eval_hr_compliance.py --split test   # test 只跑一次
```

评测数据与结果落在 `rag_qa/data/law_eval/`，体积较大，已 gitignore，按上面的脚本可以重新生成。

几组关键结果：

- 检索：全量 Recall@5 0.901，MRR 0.7919；上下文覆盖率 78.75%，拒答率 5%。
- 生成：RAGAS 忠实度 0.899、context_precision 0.713、context_recall 0.730（100 条，qwen-max 判分）。
- 快通道：词面门控后错误抢答 54 → 0，真实法律问题进入 RAG 的比例 10% → 99%。
- HR 合规：140 条 JD 评测，规则轨 P/R/F1 1.0，引用准确率 100%，干净 JD 0 误报。评测集由 30 条已有样本和 110 条按规则模板生成并逐条核对，不是全部人工标注。

## 9. 已知限制

- 多轮历史按最近 5 轮拼进提示词，跨话题追问时可能带偏检索，尚未做话题切分。
- `citation_check.py` 核对答案引用的法条是否存在，但暂不识别「第 X 条之一」这类写法，只会漏检，不会误报。
- 混合检索里 BGE-M3 稀疏向量对 top-k 排序影响较小，保留为后续优化点。
- 问答和法条以整条为单位，没有做更细的条款拆分和表格类法条处理。
- HR 合规只覆盖法律级文本，公司制度类风险仍需执行层自己的规则库。
- 13 条规则是召回上限，规则之外的歧视或违法表述可能漏检；命中项仍需人工复核。`needs_human` 只表示法条依据是否完成逐字回查，不代表法务结论。
- LLM 闭集选条使用 qwen3.6-plus，关闭思考链后单次约 1–3 秒；超时或解析失败会自动回退到确定性 hint，接口仍返回，只是 `llm_used=false`。

## 10. 文档与约定

- `docs/项目介绍.md`：项目背景、架构、流程、指标和边界；
- `docs/开发指南/README.md`：开发入口、启动和自检；
- `docs/开发指南/开发与修改指南.md`：代码结构、常见改动、守卫和排查；
- `docs/开发指南/项目现状与指标.md`：数据规模、指标和已知限制；
- `docs/开发指南/RAG项目流程图说明.md`：三阶段流程图和逐环节说明，含 Mermaid 源码；
- `docs/项目概览.html`：一页式项目概览，浏览器直接打开；
- 本文第 4～8 节：环境搭建、建库顺序、接口和评测。

命名约定：

| 项 | 命名 |
|---|---|
| 仓库 / 目录 | `lawrag` |
| 镜像 / 容器 | `lawrag:v7.5` / `lawrag-app` |
| 日志器 | `LawRAG` |
| MySQL | `law_articles`、`law_faq`、`conversations` |
| Milvus | 库 `itcast`，集合 `lawrag`、`lawfaq` |
| 知识库来源标识 | `source = "law"` |

MySQL 库名 `subjects_kg` 和 Milvus 库名 `itcast` 是早期命名，改名需要重建库并迁移数据，暂时保留。

## 免责声明

本项目为个人学习作品，语料来自公开法规原文和公开问答数据集。输出内容不构成法律意见，请勿用于真实法律决策。
