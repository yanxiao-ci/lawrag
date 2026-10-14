# -*- coding: utf-8 -*-
"""
把 RAGAS 判分产物汇总成一份结论 JSON（P2 第二步收尾用）
=======================================================
`ragas_evaluate.py` 会产出两种 CSV：
  - <out>            每格是该指标的全量取值列表（pandas 把 EvaluationResult 拍平后的样子）
  - <out>_detail.csv 逐条明细（有 question / answer 列）——有时导出失败，所以两种都要兼容

用法:
  python tools/ragas_summary.py \
      --csv rag_qa/data/law_eval/ragas/ragas_law_flash_0_50.csv \
      --csv rag_qa/data/law_eval/ragas/ragas_law_flash_50_100.csv \
      --generator qwen3.8-flash --judge qwen3.8-flash \
      --out rag_qa/data/law_eval/ragas/ragas_flash_result.json
"""
import argparse
import ast
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd                                                  # noqa: E402

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# 宽表（<out>.csv）把每个指标塞在一格字符串里，形如 {'faithfulness': 0.5, ...}
# 或 np.float64(0.95)；逐条明细（<out>_detail.csv）才是每种指标一列。两种都要能读。
PAT = {m: re.compile(r"'%s': (?:np\.float64\()?([0-9.]+|nan)\)?" % m) for m in METRICS}


def values_of(cell):
    """把一格解析成浮点列表：兼容标量、列表字符串、以及列表里的 nan。"""
    if isinstance(cell, str):
        s = cell.strip()
        items = ast.literal_eval(s) if s.startswith("[") else [s]
        if not isinstance(items, (list, tuple)):
            items = [items]
    elif isinstance(cell, (list, tuple)):
        items = cell
    else:
        items = [cell]
    out = []
    for v in items:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            out.append(f)
    return out


def load_metrics(path):
    """返回 ({metric: [分数...]}, 行数)。"""
    df = pd.read_csv(path)
    out = {m: [] for m in METRICS}
    if all(m in df.columns for m in METRICS):          # 逐条明细
        for m in METRICS:
            s = pd.to_numeric(df[m], errors="coerce").dropna()
            out[m] = [float(v) for v in s]
        return out, len(df)
    text = ""                                          # 宽表：从 scores 列里正则抠分数
    for col in ("scores", "ragas_traces"):
        if col in df.columns and df[col].notna().any():
            text = str(df[col].dropna().iloc[0])
            break
    for m, pat in PAT.items():
        for raw in pat.findall(text):
            if raw != "nan":
                out[m].append(float(raw))
    return out, len(df)


def main():
    ap = argparse.ArgumentParser(description="RAGAS 判分产物汇总")
    ap.add_argument("--csv", action="append", required=True, help="可多次传入，按批汇总")
    ap.add_argument("--out", required=True, help="结论 JSON 输出路径")
    ap.add_argument("--generator", default="", help="生成答案用的模型，写进结论便于追溯")
    ap.add_argument("--judge", default="", help="判分用的模型，写进结论便于追溯")
    args = ap.parse_args()

    collected = {m: [] for m in METRICS}
    used_rows = 0
    for path in args.csv:
        # 优先用逐条明细（<out>_detail.csv）：宽表的分数藏在字符串里，明细更好核对
        detail = path.replace(".csv", "") + "_detail.csv"
        use = detail if os.path.exists(detail) else path
        per, n_rows = load_metrics(use)
        used_rows += n_rows or 0
        for m in METRICS:
            collected[m].extend(per[m])
        print(f"  {os.path.relpath(use, ROOT)}: {n_rows} 行, "
              f"有效 {[len(per[m]) for m in METRICS]}", flush=True)

    summary = {
        "rows_in": used_rows,
        "generator_model": args.generator,
        "judge_model": args.judge,
        "n": {m: len(v) for m, v in collected.items()},
        "mean": {m: (sum(v) / len(v) if v else None) for m, v in collected.items()},
        "zero_count": {m: sum(1 for x in v if x == 0.0) for m, v in collected.items()},
        "source_csv": [os.path.relpath(p, ROOT) for p in args.csv],
    }
    # 分数保留 4 位，避免把浮点噪声当结论
    summary["mean"] = {m: (round(v, 4) if v is not None else None)
                       for m, v in summary["mean"].items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n结论已写入：{os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
