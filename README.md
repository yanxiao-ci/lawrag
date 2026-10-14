# 法律法规智能问答助手（lawrag）

一个**中文法律法规领域**的 RAG 问答系统：前面用 MySQL + Redis + BM25 做「快问快答」通道承接能精确命中的问题，
剩下的交给「法条检索 + 法律问答双召回 → 精排 → 大模型生成」的 RAG 通道，并配一套可复现的评测脚本来验证效果。

> **一个完整可运行的法律领域 RAG 系统**：法条查询可秒级回溯到原文，法律咨询由「检索增强生成」作答，
> 回答强制标注《法律名》第 X 条，并在生成后逐条做引用校验。下面所有指标都能用仓库里的脚本复算。

**亮点一览**（数字均为实测）

| 亮点 | 结果 |
| --- | --- |
| 快通道门控 | 换成词面判据后**错误抢答 54 → 0**；真实法律问题进 RAG **10% → 99%**；库内原句 **200/200** 仍秒回 |
| 双路召回 + 精排 | 法条块 2 条 + 问答块 2 条；参考条文覆盖率 **65% → 75%**，兜底拒答率 **24% → 5%** |
| 检索质量 | Recall@5 **0.824**，线上链路 MRR 0.7188（1000 条样本 + 四组消融） |
| 生成质量 | RAGAS 忠实度 **0.899**；同裁判对照上下文召回 **0.32 → 0.61**（100 条领域均衡评测集） |
| 法条可追溯 | 21,312 条法条带 法律名 / 条文号 / 编章节 / 施行日 / 时效状态，回答里的条号逐条回查上下文 |
| 时效安全 | 检索**常开** `status == "有效"` 过滤，另存 4 部历史版本作新旧对照 |

---

## 1. 能力边界（先说清楚它不做什么）

**覆盖**：全国人大及其常委会制定的「法律」原文（宪法、基本法律、法律、条例/办法等），共 **288 部现行有效法律**。

**不覆盖**（写清楚，避免夸大）：

- 行政法规、司法解释、部门规章、地方性法规、自治条例、国际条约；
- 裁判文书 / 判例 / 典型案例 / 合同模板 / 法律意见书；
- 任何形式的「个案结论」，也不构成正式法律意见。

因此它适合回答「《劳动合同法》第 X 条怎么规定」「租房押金不退可以引哪条」这类**法条依据类**问题；
对「某地最新政策是什么」「我这个案子能不能赢」只能给出法律层面的通用依据，**不能替代律师**。

---

## 2. 数据规模（实测数字，全部可用 `tools/` 脚本复算）

| 数据 | 存放位置 | 规模 |
| --- | --- | --- |
| 法律原文（.txt） | `rag_qa/data/law_data/` | **292 个文件 / 4.2 MB**（288 部现行 + 4 个历史版本） |
| 法条表 | MySQL `law_articles` | **21,565 条**（现行有效 21,312 + 历史版本 253） |
| 法条向量 | Milvus `lawrag`（子块 = 单条法条，父块 = 同法连续 5 条） | **55,847 个子块** |
| 法律问答 | MySQL `law_faq` / `mysql_qa/data/law_qa.csv` | **20,000 条** |
| 问答向量 | Milvus `lawfaq`（第三召回源） | **20,000 条** |

数据来源：

- **法条主料**：[KuugoRen/Chinese_Law](https://modelscope.cn/datasets/KuugoRen/Chinese_Law)（97 部法律 .txt）
  → 用 `tools/expand_law_corpus.py` 从本地全量法规库按「法律」级别扩容到 292 部；
- **法律问答**：`Brench/chinese_law_data_rag_ft`（HF 镜像，5 万条测试集）
  → `tools/sample_law_faq.py` 按「领域均衡 + 答案引用的法条必须真实存在」抽 2 万条；
- **检索评测集**：[ByronLeeee/CN-Law-Query-Retrieval-Dataset](https://modelscope.cn/datasets/ByronLeeee/CN-Law-Query-Retrieval-Dataset)（anchor/positive/negative）；
- **编 / 章 / 节层级**：`Dusker/chinese-laws-pretrain`（用于补 `chapter` 元数据，可选）。

采集脚本：`python tools/collect_law_data.py`（HF 直连不通时默认走 `hf-mirror.com`，可用 `HF_ENDPOINT` 覆盖）。

---

## 3. 系统架构

完整流程图见 [`docs/RAG项目流程图说明.md`](docs/RAG项目流程图说明.md)（源文件 `docs/make_flowchart.py`，输出到 `docs/figures/`）。

总体分三段：

**A · 离线准备（建库，一次性）**

1. `rag_qa/data/law_data/*.txt` → 按**法条边界**切分（子块 = 一条完整法条，父块 = 同一部法律连续 5 条，
   父子关系写入元数据），避免通用递归切分把法条拦腰截断导致引用不准；
2. BGE-M3 向量化（稠密 1024 维 + 稀疏向量）→ 写入 Milvus 集合 `lawrag`；
3. 法条结构化信息 → MySQL `law_articles`（含编/章/节、公布日、施行日、时效状态）；
4. 2 万条法律问答 → MySQL `law_faq` + Redis 问题列表 → BM25 检索器就绪；
5. 2 万条问答向量化 → Milvus 集合 `lawfaq`（RAG 的第三召回源）。

**B · 在线问答（每次请求）**

```
用户提问
  ├─ 问候语正则 → 模板回复（不检索）
  ├─ 法条号精确路由（“第 X 条”+ 法律名）→ 直接查 law_articles
  ├─ ① 快问快答通道：Redis 缓存 → Jieba → BM25 → 词面门控（Jaccard / 查询词覆盖率 ≥ 0.85）→ 命中即回标准答案
  └─ ② RAG 通道：BERT 意图识别 → 策略选择（直接/假设 HyDE/子查询/回溯）
                 → 法条 + 问答双路召回 → BGE-Reranker 精排 → 取上下文
                 → 拼接提示词 → 大模型流式生成 → 引用校验
```

**C · 评估迭代**

- 检索侧：`tools/eval_recall.py`（Recall@k / MRR，支持消融对比）；
- 生成侧：RAGAS 四项指标（忠实度 / 答案相关性 / 上下文相关性 / 上下文召回率）；
- 门控侧：`tools/tune_fast_channel_gate.py`（快通道抢答准确率标定）；
- 上下文覆盖：`tools/eval_context_coverage.py`（不依赖 LLM，直接量上下文里有没有关键条文）。

---

## 4. 快速开始

### 4.1 依赖服务

需要三个外部服务（本项目跑在虚拟机上，用 Docker 部署）：

| 服务 | 默认地址 | 说明 |
| --- | --- | --- |
| MySQL 8 | `192.168.229.128:3307` | 库 `subjects_kg`，表 `law_articles` / `law_faq` / 会话历史 |
| Redis | `192.168.229.128:6379` | 快通道缓存与会话缓存 |
| Milvus | `192.168.229.128:19530` | 库 `itcast`，集合 `lawrag` / `lawfaq` |

`docker-compose.yml` 只包含应用本身（`lawrag-app`），三个依赖服务需自行部署（Milvus standalone 需要 etcd + minio）。

### 4.2 Python 环境

```bash
conda create -n lawrag python=3.10 -y
conda activate lawrag
pip install -r requirements.txt
```

> 装 `torch` 请按显卡 CUDA 版本选官方 wheel；有 GPU 时向量化与精排会自动用上（`vector_store.py` 里做了设备探测）。

### 4.3 模型（不入库，需要自备）

权重体积过大，已在 `.gitignore` 中排除，请按下面的目录名放到 `rag_qa/models/`：

| 目录 | 用途 | 获取方式 |
| --- | --- | --- |
| `bert-base-chinese` | 意图分类器的底座 | HuggingFace / ModelScope 下载 |
| `bge-m3` | 稠密 + 稀疏向量模型 | ModelScope `BAAI/bge-m3` |
| `bge-reranker-large` | 精排模型（CrossEncoder） | ModelScope `BAAI/bge-reranker-large` |
| `bert_query_classifier_law` | 意图分类（通用知识 / 专业咨询） | 用 `python tools/train_law_classifier.py` 训练生成 |

### 4.4 配置

配置优先级：**`.env` > `config.ini` > 代码内 fallback**。`.env` 已排除在 git 之外，本地新建即可：

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

# 检索参数
RETRIEVAL_K=5
CANDIDATE_M=2

VALID_SOURCES=["law"]
CUSTOMER_SERVICE_PHONE=13000000000
```

### 4.5 建库（顺序不能换）

```bash
# 1) 法条 + 问答入库 MySQL（会 TRUNCATE 后重写 law_articles / law_faq）
python tools/import_law_data.py

# 2) 法条向量库：语料换过必须加 --reset，否则新旧语料块会混在同一个集合里
python tools/build_law_index.py --reset

# 3) 问答向量库（RAG 第三召回源）
python tools/build_lawfaq_index.py --reset
```

### 4.6 启动

```bash
python app.py          # 或 uvicorn app:app --host 0.0.0.0 --port 8080
```

浏览器打开 <http://127.0.0.1:8080/>（注意不是 `0.0.0.0`，那是监听地址）。

接口一览：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/query` | 一次性问答 |
| WS | `/api/stream` | 流式问答（前端默认走这个） |
| POST | `/api/create_session` | 新建会话 |
| GET/DELETE | `/api/history/{session_id}` | 查询 / 清空会话历史 |
| GET | `/health` | 存活检查 |
| GET | `/api/sources` | 可用知识库来源（当前为 `["law"]`） |

---

## 5. 目录结构

```
lawrag/
├─ app.py                     # FastAPI 入口（HTTP + WebSocket 流式）
├─ new_main.py                # 在线问答总调度：问候语 → 条文号路由 → 快通道 → RAG
├─ config.ini / .env          # 配置（.env 覆盖 config.ini，已 gitignore）
├─ base/                      # 配置、日志、法律语料解析器（law_parser.py）
├─ mysql_qa/                  # 快问快答通道：MySQL / Redis / BM25 检索与门控
├─ rag_qa/
│  ├─ core/                   # 意图识别、策略选择、向量库、检索、提示词、引用校验
│  ├─ document_loaders/       # PDF/Word/PPT/图片 加载器 + OCR
│  ├─ text_splitters/         # 法条边界切分器（law_article_splitter.py）+ 通用中文切分
│  ├─ models/                 # 模型权重（不入库）
│  ├─ data/                   # 法律原文、分类数据、评测产物
│  └─ rag_assessment/         # RAGAS 评测脚本
├─ static/index.html          # 前端页面（流式对话 + 引用展示）
├─ tools/                     # 数据采集 / 建库 / 评测 / 调参脚本
└─ docs/                      # 方案书、流程图与迭代记录
```

> 早期文件（旧入口、旧前端、旧语料、旧评测集）已移到根目录 `_archive/` 并在 `.gitignore` 中排除；
> 仓库主体只保留法律版，这些文件仍在本机与 git 历史里，可随时找回。

---

## 6. 评测怎么跑

```bash
# 检索质量：Recall@k / MRR，含 dense / hybrid / +rerank 消融
python tools/eval_recall.py

# 上下文覆盖率（不依赖 LLM，看召回的上下文里有没有关键条文）
python tools/eval_context_coverage.py

# 快通道门控标定（库内原句 vs 真实法律问题，看抢答准确率）
python tools/tune_fast_channel_gate.py

# 生成质量（RAGAS 四项指标）：先建评测集 → 跑 RAG 通道收答案 → 判分
python tools/build_law_ragas_set.py
python tools/collect_law_ragas_answers.py --workers 8
python rag_qa/rag_assessment/ragas_evaluate.py --data rag_qa/data/law_eval/ragas/law_ragas_set.json --out rag_qa/data/law_eval/ragas/ragas_law.csv
```

评测数据与结果落在 `rag_qa/data/law_eval/`（体积较大，已 gitignore，按上面的脚本可重新生成）。

---

## 7. 已知限制与后续

- **多轮历史污染**：会话历史按「最近 5 轮」拼进提示词，跨话题追问时可能带偏检索，尚未做话题切分；
- **引用校验**：`citation_check.py` 会核对答案引用的法条是否存在，但暂不识别「第 X 条之一」这类写法（只会漏检，不会误报）；
- **稀疏分支**：混合检索里 BGE-M3 稀疏向量当前对 top-k 排序影响很小（见 §6 消融），保留为后续优化点；
- **粒度**：问答与法条目前是「整条法条 / 整条问答」为最小单位，未做更细的条款拆分与表格类法条处理。

---

## 8. 文档索引

- [`docs/RAG项目流程图说明.md`](docs/RAG项目流程图说明.md)：三阶段流程图与逐环节说明（含 Mermaid 源码，图源文件 `docs/make_flowchart.py`）；
- 本文 §4～§6：环境搭建、建库顺序、评测怎么跑。

> 数据规模评估与系统设计这两份文档属于作者本机资料，不随仓库发布；代码注释里提到的「设计文档 §X」指的就是它们。

---

## 9. 免责声明

本项目为个人学习作品，语料为公开法规原文与公开问答数据集，**输出内容不构成法律意见**，请勿用于真实法律决策。

---

## 10. 命名约定

| 项 | 命名 |
| --- | --- |
| 仓库 / 目录 | `lawrag` |
| 镜像 / 容器 | `lawrag:v7.5` / `lawrag-app` |
| 日志器 | `LawRAG` |
| MySQL | 表 `law_articles`（法条）、`law_faq`（法律问答）、`conversations`（会话） |
| Milvus | 库 `itcast`，集合 `lawrag`（法条向量）、`lawfaq`（问答向量） |
| 知识库来源标识 | `source = "law"`（前端显示为「来源」） |

代码里早期遗留的旧命名已统一清理：文档加载器与文本切分器分别归到 `rag_qa/document_loaders/`
与 `rag_qa/text_splitters/`，表列名统一为 `law_faq` / `domain`。
仍保留的两个早期库名是 MySQL `subjects_kg` 与 Milvus `itcast`（改名需要重建库并迁移数据，暂不动）。
