# -*- coding: utf-8 -*-
"""
构建劳动法 RAGAS 生成评测集
===========================
从 2,043 条「劳动法务」FAQ 里抽 100 条，要求答案引用的条号能在当前语料里查到，
产出 collect_law_ragas_answers.py / ragas_evaluate.py 可直接吃的 JSON。

用法:
  python tools/build_hr_ragas_set.py --n 100
  python tools/build_hr_ragas_set.py --n 100 --seed 20261021
"""

import argparse
import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.domains import law_names_for  # noqa: E402
from base.law_parser import norm_law_name, parse_law_dir  # noqa: E402
from rag_qa.core.citation_check import extract_citations  # noqa: E402

FAQ_CSV = os.path.join(ROOT, "mysql_qa", "data", "law_qa.csv")
LAW_DIR = os.path.join(ROOT, "rag_qa", "data", "law_data")
OUT_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "hr_eval", "ragas", "hr_ragas_set.json")


def main():
    ap = argparse.ArgumentParser(description="构建劳动法 RAGAS 评测集")
    ap.add_argument("--faq", default=FAQ_CSV)
    ap.add_argument("--law-dir", default=LAW_DIR)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--domain", default="劳动法务")
    ap.add_argument("--profile", default="labor")
    ap.add_argument("--allow-out-of-profile", action="store_true",
                    help="允许引用领域外法律（默认不允许，保证 strict 模式可答）")
    args = ap.parse_args()

    print("解析当前语料 ...")
    corpus = {(norm_law_name(law), num) for law, _no, num, _text in parse_law_dir(args.law_dir)}
    profile_laws = {norm_law_name(x) for x in law_names_for(args.profile)}
    print(f"语料条号 {len(corpus):,} 个；领域 {args.profile} {len(profile_laws)} 部法")

    pool = []
    total = 0
    with open(args.faq, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if row.get("法律领域") != args.domain:
                continue
            q = (row.get("问题") or "").strip()
            a = (row.get("答案") or "").strip()
            if len(q) < 5 or len(a) < 20:
                continue
            cites = extract_citations(a)
            if not cites or not cites.issubset(corpus):
                continue
            if not args.allow_out_of_profile and any(law not in profile_laws for law, _num in cites):
                continue
            pool.append({
                "question": q,
                "ground_truth": a,
                "domain": args.domain,
                "cited_laws": sorted({law for law, _num in cites}),
                "source": "law_faq",
            })

    print(f"FAQ 总数 {total:,}；{args.domain} 且引用可核对 {len(pool):,} 条")
    if len(pool) < args.n:
        raise SystemExit(f"可用样本不足：需要 {args.n}，只有 {len(pool)}")

    random.Random(args.seed).shuffle(pool)
    items = pool[:args.n]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    law_counts = {}
    for it in items:
        for law in it["cited_laws"]:
            law_counts[law] = law_counts.get(law, 0) + 1
    print(f"写出 {len(items)} 条 -> {os.path.relpath(args.out, ROOT)}")
    print("引用法律分布：" + " · ".join(
        f"{k} {v}" for k, v in sorted(law_counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
