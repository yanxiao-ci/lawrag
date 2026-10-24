"""
    百炼千问模型
"""
# 导入 argparse，用于指定评测数据文件（默认用法律版评测集收集出的答案文件）
import argparse
# 导入 os，用于把相对路径解析成相对脚本所在目录
import os
# 导入 sys，把项目根目录放进模块搜索路径（否则命令行直接跑找不到 base/rag_qa 包）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入 RunConfig：控制评测并发与超时。
# 免费额度下默认 16 并发会大量超时（实测 2026-10-14：qwen3.8-flash 那轮 50 条里
# 后半段整片 TimeoutError、判分变 nan），所以并发数做成了命令行参数。
from ragas import RunConfig
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
from ragas.metrics import (
    faithfulness,  # 忠实度
    answer_relevancy,  # 答案相关性
    context_precision,  # 上下文相关性
    context_recall # 上下文召回率
)
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入langchain_openai的嵌入模型和聊天模型，用于评估时的语义计算和推理
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
# 导入json库，用于加载JSON格式的评估数据集
import json
from base.config import config

# 脚本目录：默认数据/结果路径都相对这里，换工作目录也能跑
DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="RAGAS 评测（支持换数据集）")
LAW_EVAL_DIR = os.path.normpath(os.path.join(DIR, "..", "data", "law_eval", "ragas"))
parser.add_argument("--data", default=os.path.join(LAW_EVAL_DIR, "law_ragas_answers.json"),
                    help="评测数据：JSON 列表，每条含 question/contexts/answer/ground_truth")
parser.add_argument("--out", default=os.path.join(LAW_EVAL_DIR, "ragas_law.csv"),
                    help="结果 CSV 输出路径")
parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（调试用，0=全部）")
parser.add_argument("--start", type=int, default=0, help="从第几条开始（分批跑用）")
# 评评测并发与超时：judge 模型额度紧张/响应慢时，把 --workers 调小
parser.add_argument("--workers", type=int, default=4, help="评测并发数（默认 4，免费额度下别调太高）")
parser.add_argument("--timeout", type=int, default=240, help="单次调用超时秒数（默认 240）")
# 判分模型独立于生成模型：某模型额度用尽时只换判分模型，不动生产链路配置
parser.add_argument("--judge-model", default=config.LLM_MODEL,
                    help="判分模型名（默认取 config.LLM_MODEL；额度用尽时换百炼上可用的新模型）")
# 只跑部分指标（answer_relevancy 在免费额度/中文场景下不稳定，可按需关掉）
parser.add_argument("--metrics", default="faithfulness,answer_relevancy,context_precision,context_recall",
                    help="逗号分隔的指标名，可选 faithfulness/answer_relevancy/context_precision/context_recall")
args = parser.parse_args()
print(f"评测数据：{args.data}")

# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open(args.data, "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)
if args.limit or args.start:
    data = data[args.start:(args.start + args.limit) if args.limit else None]


def ctx_of(item):
    """兼容两种字段名：老数据用 context，新收集的用 contexts。"""
    return item.get("context") or item.get("contexts") or []

# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [ctx_of(item) for item in data],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data]
}
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)

# 3. 配置RAGAS评估环境
# # 初始化ChatOpenAI模型，指定模型，并设置OpenAI API密钥
llm = ChatOpenAI(
    model_name=args.judge_model,
    openai_api_key=config.DASHSCOPE_API_KEY,
    openai_api_base=config.DASHSCOPE_BASE_URL,
    temperature=0.0)
# 初始化OpenAI嵌入模型，用于计算语义相似度，设置API密钥
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=config.DASHSCOPE_API_KEY)

# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
METRIC_MAP = {
    "faithfulness": faithfulness,          # 忠实度：答案是否基于上下文
    "answer_relevancy": answer_relevancy,  # 答案相关性：答案与问题的匹配度
    "context_precision": context_precision,# 上下文相关性：上下文是否仅包含相关信息
    "context_recall": context_recall,      # 上下文召回率：上下文是否包含所有必要信息
}
metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
unknown = [m for m in metric_names if m not in METRIC_MAP]
if unknown:
    raise SystemExit(f"未知指标：{unknown}，可选：{list(METRIC_MAP)}")
print(f"判分模型：{args.judge_model} | 指标：{metric_names} | 并发：{args.workers}", flush=True)

result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[METRIC_MAP[m] for m in metric_names],
    # 传入配置好的LLM模型
    llm=llm,
    # 传入配置好的嵌入模型
    embeddings=embeddings,
    # 并发与超时（命令行可调，避免额度紧张时整片超时把分数打成 nan）
    run_config=RunConfig(max_workers=args.workers, timeout=args.timeout)
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)
# 将评估结果转换为pandas DataFrame，便于保存
result_df = pd.DataFrame([result])
# 将DataFrame保存为CSV文件，文件名为ragas_evaluation_results.csv，不保存索引
result_df.to_csv(args.out, index=False)
print(f"结果已写入：{args.out}")

# 逐条明细（便于复核低分是哪几条、为什么低）：<结果名>_detail.csv
try:
    detail = result.to_pandas()
    detail_path = args.out.replace(".csv", "") + "_detail.csv"
    detail.to_csv(detail_path, index=False)
    print(f"逐条明细已写入：{detail_path}")
except Exception as e:
    print(f"逐条明细导出失败（不影响上面的总分）：{e}")
