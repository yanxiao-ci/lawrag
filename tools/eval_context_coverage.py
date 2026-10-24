# -*- coding: utf-8 -*-
"""
上下文覆盖率评测（不依赖 LLM）
=============================
用法律版评测集（build_law_ragas_set.py 产物）衡量：**喂给 LLM 的上下文里，
参考答案引用的法条覆盖了多少**。这是 RAGAS context_recall 的无 LLM 代理指标，
评测集里每条参考答案都带《X法》第Y条 引用且已核对存在于语料。

用途：量化「注入几条上下文（CANDIDATE_M）」这个杠杆的收益。
线上默认 RETRIEVAL_K=5（粗排候选）、CANDIDATE_M=2（注入条数）。

用法:
  python tools/eval_context_coverage.py --topn 2
  python tools/eval_context_coverage.py --topn 5 --k 10
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SET_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas", "law_ragas_set.json")

ap = argparse.ArgumentParser(description="上下文覆盖率（法条级）")
ap.add_argument("--data", default=SET_DEFAULT)
ap.add_argument("--topn", type=int, default=2, help="注入上下文条数 = CANDIDATE_M")
ap.add_argument("--k", type=int, default=5, help="粗排候选数 = RETRIEVAL_K")
ap.add_argument("--strategy", default="直接检索")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--domain", default=None, help="领域视图名（如 labor）；默认不过滤")
ap.add_argument("--domain-mode", default="off", choices=["off", "soft", "strict"])
args = ap.parse_args()

# 必须在 import config/vector_store 之前设进环境变量：config 是导入时读取的
os.environ["CANDIDATE_M"] = str(args.topn)
os.environ["RETRIEVAL_K"] = str(args.k)

from base.config import config                                    # noqa: E402
from rag_qa.core.citation_check import extract_citations          # noqa: E402
from rag_qa.core.new_rag_system import RAGSystem                  # noqa: E402
from rag_qa.core.vector_store import VectorStore                  # noqa: E402


def main():
    items = json.load(open(args.data, encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]
    print(f"评测集 {len(items)} 条 | RETRIEVAL_K={config.RETRIEVAL_K} "
          f"CANDIDATE_M={config.CANDIDATE_M} | 策略={args.strategy}", flush=True)

    rag = RAGSystem(VectorStore(), llm=lambda prompt: iter(()))   # 只做检索，不出答案
    n = all_hit = some_hit = empty_ctx = 0
    ratios, ctx_counts, ctx_chars = [], [], []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        want = extract_citations(it["ground_truth"])
        if not want:
            continue
        docs = rag.retrieve_and_merge(
            it["question"], source_filter="law", strategy=args.strategy,
            domain=args.domain, domain_mode=args.domain_mode)
        contexts = [rag._format_context_doc(d) for d in docs] if docs else []
        n += 1
        empty_ctx += (not contexts)
        ctx_counts.append(len(contexts))
        ctx_chars.append(sum(len(c) for c in contexts))
        hit = len(want & extract_citations("\n\n".join(contexts)))
        ratios.append(hit / len(want))
        all_hit += (hit == len(want))
        some_hit += (hit > 0)
        if i % 25 == 0:
            print(f"  {i}/{len(items)} 全命中={all_hit}/{n} ({time.time() - t0:.0f}s)", flush=True)

    res = {
        "n": n, "topn": args.topn, "k": args.k, "strategy": args.strategy,
        "all_cited_covered": round(all_hit / n, 4),
        "some_cited_covered": round(some_hit / n, 4),
        "avg_coverage": round(sum(ratios) / n, 4),
        "avg_contexts": round(sum(ctx_counts) / n, 2),
        "avg_context_chars": int(sum(ctx_chars) / n),
        "empty_context": empty_ctx,
    }
    print("\n" + json.dumps(res, ensure_ascii=False), flush=True)
    prefix = f"coverage_{args.domain}_{args.domain_mode}_" if args.domain else "coverage_"
    out = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas",
                       f"{prefix}topn{args.topn}_k{args.k}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已存：{os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
