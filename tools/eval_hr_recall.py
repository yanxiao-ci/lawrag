# -*- coding: utf-8 -*-
"""
劳动法领域检索评测：off / soft / strict 三档对比
================================================
复用 eval_recall.py 的口径（目标条号命中、父块去重、同一精排模型），
额外量三个领域指标：
  - cross_domain_rate：召回的父块中不属于领域法名的比例
  - fallback_rate：soft 模式因命中不足而放开过滤的比例
  - 拒答率：top-k 里完全没有目标条号的样本比例（= 1 - Recall@k）

用法:
  python tools/eval_hr_recall.py --data rag_qa/data/hr_eval/retrieval_hr.jsonl
  python tools/eval_hr_recall.py --data ... --modes off soft strict --k 5
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.documents import Document  # noqa: E402

from base.config import config  # noqa: E402
from base.domains import law_names_for  # noqa: E402
from base.law_parser import norm_law_name  # noqa: E402
from rag_qa.core.citation_check import extract_citations  # noqa: E402
from rag_qa.core.vector_store import VectorStore  # noqa: E402

DATA_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "hr_eval", "retrieval_hr.jsonl")
OUT_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "hr_eval", "eval_hr_recall_result.json")


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parents_with_meta(chunks):
    """按 parent_content 去重成父块，保留 metadata（算 cross_domain_rate 要用 law_name）。"""
    seen, out = set(), []
    for c in chunks:
        key = c.metadata.get("parent_content") or c.page_content
        if key in seen:
            continue
        seen.add(key)
        out.append(Document(page_content=key, metadata=dict(c.metadata)))
    return out


def retrieve(vs, query, k, law_names):
    chunks = vs.search_children(query, k=k, source_filter="law", law_names=law_names)
    parents = parents_with_meta(chunks)
    if len(parents) >= 2:
        scores = vs.reranker.predict([[query, d.page_content] for d in parents])
        parents = [d for _, d in sorted(zip(scores, parents), reverse=True)]
    return parents[:k]


def main():
    ap = argparse.ArgumentParser(description="劳动法领域检索评测")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--profile", default="labor")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["off", "soft", "strict"],
                    choices=["off", "soft", "strict"])
    args = ap.parse_args()

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit(f"评测集为空: {args.data}")
    laws = law_names_for(args.profile)
    if not laws:
        raise SystemExit(f"领域 {args.profile} 没有配置法名")
    norm_laws = {norm_law_name(x) for x in laws}
    print(f"评测集 {len(rows)} 条；领域 {args.profile} {len(laws)} 部法；k={args.k}")

    vs = VectorStore()
    ks = [x for x in (1, 3, 5) if x <= args.k]
    results = {}

    for mode in args.modes:
        law_names = laws if mode in ("soft", "strict") else None
        hit = {kk: 0 for kk in ks}
        mrr = 0.0
        cross = 0
        retrieved = 0
        fallback = 0
        t0 = time.time()
        for i, r in enumerate(rows, 1):
            q = r["query"]
            docs = retrieve(vs, q, args.k, law_names)
            if mode == "soft" and len(docs) < config.CANDIDATE_M:
                extra = retrieve(vs, q, args.k, None)
                seen = {d.page_content for d in docs}
                for d in extra:
                    if d.page_content not in seen:
                        docs.append(d)
                        seen.add(d.page_content)
                fallback += 1

            target = (norm_law_name(r["target_law"]), r["target_article_num"])
            rank = 0
            for pos, d in enumerate(docs, 1):
                if target in extract_citations(d.page_content):
                    rank = pos
                    break
            if rank:
                mrr += 1.0 / rank
                for kk in ks:
                    if rank <= kk:
                        hit[kk] += 1

            for d in docs:
                retrieved += 1
                ln = norm_law_name(d.metadata.get("law_name", ""))
                if ln and ln not in norm_laws:
                    cross += 1

            if i % 50 == 0:
                print(f"  [{mode}] {i}/{len(rows)}  Recall@{args.k}={hit[args.k] / i:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        n = len(rows)
        results[mode] = {
            "n": n,
            "seconds": round(time.time() - t0, 1),
            "mrr": round(mrr / n, 4),
            **{f"recall@{kk}": round(hit[kk] / n, 4) for kk in ks},
            "cross_domain_rate": round(cross / retrieved, 4) if retrieved else 0.0,
            "fallback_rate": round(fallback / n, 4),
            "abstain_rate": round(1 - hit[ks[-1]] / n, 4),
        }
        print(f"  [{mode}] {json.dumps(results[mode], ensure_ascii=False)}")

    print("\n| 模式 | " + " | ".join(f"Recall@{kk}" for kk in ks) +
          " | MRR | 跨领域率 | 兜底率 | 拒答率 | 耗时 |")
    print("|---" * (len(ks) + 6) + "|")
    for mode, r in results.items():
        cells = " | ".join(f"{r[f'recall@{kk}']:.4f}" for kk in ks)
        print(f"| {mode} | {cells} | {r['mrr']:.4f} | {r['cross_domain_rate']:.2%} | "
              f"{r['fallback_rate']:.2%} | {r['abstain_rate']:.2%} | {r['seconds']}s |")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"data": args.data, "profile": args.profile, "k": args.k, "results": results},
                  f, ensure_ascii=False, indent=1)
    print(f"\n结果已存：{os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
