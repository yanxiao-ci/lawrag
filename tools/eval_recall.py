# -*- coding: utf-8 -*-
"""
检索量化评测：Recall@k / MRR + 消融（设计文档§5.4 / §7 步骤 7）
====================================================================
数据：{anchor, positive, negative} 三元组（ByronLeeee，已裁成语料内子集），
     衡量「给定问题能否召回正确法条」——正例的 title 就是你期望召回的那一条。

评测对象是**父块**（喂给 LLM 的上下文就是父块），不是子块。
消融组合见 MODES：
  dense                  只走向量（稠密），无精排
  hybrid                 稠密 + BGE-M3 稀疏，无精排        ← 线上粗排
  hybrid_rerank          稠密 + 稀疏 + bge-reranker 精排     ← 线上完整链路
  hybrid_nofilter        同 hybrid，但不过滤历史版本（对照）
  hybrid_rerank_nofilter 同 hybrid_rerank，但不过滤历史版本（对照）

用法:
  python tools/eval_recall.py --limit 300
  python tools/eval_recall.py --limit 300 --modes dense hybrid hybrid_rerank hybrid_rerank_nofilter
"""
import argparse
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.documents import Document              # noqa: E402

from base.law_parser import norm_law_name, cn2int          # noqa: E402
from mysql_qa.db.mysql_client import MySQLClient           # noqa: E402
from rag_qa.core.citation_check import extract_citations   # noqa: E402
from rag_qa.core.vector_store import VectorStore           # noqa: E402

EVAL_FILE = os.path.join(ROOT, "rag_qa", "data", "law_eval", "retrieval",
                         "finetune_dataset_incorpus.jsonl")
RESULT_FILE = os.path.join(ROOT, "rag_qa", "data", "law_eval", "retrieval",
                           "eval_recall_result.json")
TITLE_RE = re.compile(r"title:\s*(.+?)\s*第([一二三四五六七八九十百千万零〇两0-9]+)条\s*\|")

# 消融组合：use_sparse=走不走稀疏分支 / rerank=走不走精排 / effective=只取现行有效
MODES = {
    "dense":                  dict(use_sparse=False, rerank=False, effective=True),
    "dense_rerank":           dict(use_sparse=False, rerank=True,  effective=True),
    "hybrid":                 dict(use_sparse=True,  rerank=False, effective=True),
    "hybrid_rerank":          dict(use_sparse=True,  rerank=True,  effective=True),
    "hybrid_nofilter":        dict(use_sparse=True,  rerank=False, effective=False),
    "hybrid_rerank_nofilter": dict(use_sparse=True,  rerank=True,  effective=False),
}


def target_of(positive):
    """从 positive 的 'title: X 第Y条 | text: ...' 抽取目标 (法律名, 条文号整型)。"""
    m = TITLE_RE.search(positive or "")
    if not m:
        return None
    num = cn2int(m.group(2))
    if num is None:
        return None
    return (norm_law_name(m.group(1)), num)


def ordered_parent_docs(chunks):
    """按粗排顺序把子块去重成父块（父块才是喂给 LLM 的上下文）。"""
    seen, out = set(), []
    for c in chunks:
        key = c.metadata.get("parent_content") or c.page_content
        if key in seen:
            continue
        seen.add(key)
        out.append(Document(page_content=key))
    return out


def retrieve_parents(vs, query, k, cfg):
    """按消融组合检索，返回排好序的父块列表。"""
    chunks = vs.search_children(query, k=k, source_filter="law",
                                effective_only=cfg["effective"], use_sparse=cfg["use_sparse"])
    parents = ordered_parent_docs(chunks)
    # 精排：和 hybrid_search_with_rerank 用的是同一个模型、同一套打分排序
    if cfg["rerank"] and len(parents) >= 2:
        scores = vs.reranker.predict([[query, d.page_content] for d in parents])
        parents = [d for _, d in sorted(zip(scores, parents), reverse=True)]
    return parents[:k]


def main():
    ap = argparse.ArgumentParser(description="法条检索 Recall@k / MRR + 消融")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["hybrid", "hybrid_rerank"],
                    choices=list(MODES))
    ap.add_argument("--all-samples", action="store_true",
                    help="不过滤「目标不在库里」的样本（默认过滤，否则它们必然是未命中）")
    args = ap.parse_args()

    # 1) 读样本，抽出目标法条
    triples = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            t = target_of(o.get("positive", ""))
            if t:
                triples.append((o["anchor"], t))
    random.Random(args.seed).shuffle(triples)

    # 2) 可达性过滤：目标必须能在 law_articles 里查到，否则线上怎么改都召回不到
    mysql = MySQLClient()
    mysql.cursor.execute("SELECT law_name, article_num, status FROM law_articles")
    effective = {(norm_law_name(law), num)
                 for law, num, status in mysql.cursor.fetchall() if status == "有效"}
    reachable = [(q, t) for q, t in triples if t in effective]
    samples = (triples if args.all_samples else reachable)[:args.limit]
    print(f"样本：可用 {len(triples)}，指向现行有效条文 {len(reachable)}，"
          f"本轮取 {len(samples)} 条，top-{args.k}")

    # 3) 逐消融组合跑（模型只加载一次）
    vs = VectorStore()
    ks = [x for x in (1, 3, 5) if x <= args.k]
    results = {}
    for mode in args.modes:
        cfg = MODES[mode]
        hit = {kk: 0 for kk in ks}
        mrr = 0.0
        t0 = time.time()
        for i, (query, target) in enumerate(samples, 1):
            docs = retrieve_parents(vs, query, args.k, cfg)
            rank = 0
            for r, doc in enumerate(docs, 1):
                if target in extract_citations(doc.page_content):
                    rank = r
                    break
            if rank:
                mrr += 1.0 / rank
                for kk in ks:
                    if rank <= kk:
                        hit[kk] += 1
            if i % 50 == 0:
                print(f"  [{mode}] {i}/{len(samples)} "
                      f"Recall@{args.k}={hit[args.k] / i:.3f} ({time.time() - t0:.0f}s)", flush=True)
        n = len(samples)
        results[mode] = {"n": n, "seconds": round(time.time() - t0, 1),
                         "mrr": round(mrr / n, 4)}
        results[mode].update({f"recall@{kk}": round(hit[kk] / n, 4) for kk in ks})
        print(f"  [{mode}] 完成：{json.dumps(results[mode], ensure_ascii=False)}", flush=True)

    # 4) 汇总
    print("\n| 组合 | " + " | ".join(f"Recall@{kk}" for kk in ks) + " | MRR | 耗时 |")
    print("|---" * (len(ks) + 3) + "|")
    for mode, r in results.items():
        cells = " | ".join(f"{r[f'recall@{kk}']:.4f}" for kk in ks)
        print(f"| {mode} | {cells} | {r['mrr']:.4f} | {r['seconds']}s |")
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({"limit": args.limit, "k": args.k, "seed": args.seed,
                   "modes": args.modes, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n结果已存：{os.path.relpath(RESULT_FILE, ROOT)}")


if __name__ == "__main__":
    main()
