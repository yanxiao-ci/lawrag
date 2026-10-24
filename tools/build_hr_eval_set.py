# -*- coding: utf-8 -*-
"""
构建劳动法领域的检索评测集
==========================
从现有 63,311 条 in-corpus 三元组里，筛出 positive 属于劳动法领域的样本，
固定 seed 抽样，产出 eval_hr_recall.py 可直接吃的 JSONL。

用法:
  python tools/build_hr_eval_set.py --limit 400
  python tools/build_hr_eval_set.py --profile labor --seed 20261021
"""

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.domains import get_domain, law_names_for  # noqa: E402
from base.law_parser import cn2int, norm_law_name  # noqa: E402

SRC_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "law_eval", "retrieval",
                           "finetune_dataset_incorpus.jsonl")
OUT_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "hr_eval", "retrieval_hr.jsonl")
TITLE_RE = re.compile(r"title:\s*(.+?)\s*第([一二三四五六七八九十百千万零〇两0-9]+)条\s*\|")


def target_of(positive):
    m = TITLE_RE.search(positive or "")
    if not m:
        return None
    num = cn2int(m.group(2))
    if num is None:
        return None
    return m.group(1).strip(), f"第{m.group(2)}条", num


def main():
    ap = argparse.ArgumentParser(description="构建劳动法检索评测集")
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--profile", default="labor")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--per-law-cap", type=int, default=40,
                    help="每部法最多抽多少条；避免民法典这类大法淹没劳动法样本")
    ap.add_argument("--core-only", action="store_true",
                    help="只取核心劳动法（不含个人信息保护法/民法典等补充法）")
    args = ap.parse_args()

    profile = get_domain(args.profile) or {}
    if args.core_only:
        laws = list(profile.get("laws") or [])
    else:
        laws = law_names_for(args.profile)
    if not laws:
        raise SystemExit(f"领域 {args.profile} 没有配置法名，检查 domain_profiles.json")
    norm_laws = {norm_law_name(x) for x in laws}
    print(f"领域 {args.profile}: {len(laws)} 部法（core_only={args.core_only}）")

    rows = []
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            t = target_of(o.get("positive", ""))
            if not t:
                continue
            law, article_no, num = t
            if norm_law_name(law) not in norm_laws:
                continue
            rows.append({
                "query": o.get("anchor", ""),
                "target_law": norm_law_name(law),
                "target_article": article_no,
                "target_article_num": num,
                "positive": o.get("positive", ""),
                "negative": o.get("negative", ""),
            })

    # 按法律分层抽样：先每部法各自打乱并截断，再合并打乱，避免大法淹没小法
    rng = random.Random(args.seed)
    by_law = {}
    for r in rows:
        by_law.setdefault(r["target_law"], []).append(r)
    picked = []
    for law, items in by_law.items():
        rng.shuffle(items)
        picked.extend(items[:args.per_law_cap])
    rng.shuffle(picked)
    rows = picked[:args.limit]
    if not rows:
        raise SystemExit("没有筛到样本；检查领域法名与三元组 title 格式")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in rows:
        counts[r["target_law"]] = counts.get(r["target_law"], 0) + 1
    print(f"写出 {len(rows)} 条 -> {os.path.relpath(args.out, ROOT)}")
    print("按法律分布：" + " · ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
