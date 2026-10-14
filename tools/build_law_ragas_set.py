# -*- coding: utf-8 -*-
"""
法律版 RAGAS 评测集：问题 + 参考答案（P2 第二步）
================================================
来源：`laws_data_50k_test.jsonl`（法律问答，答案自带《X法》第Y条 引用）。
筛选（沿用 sample_law_faq.py 的判据）：
  1) 答案引用的法律必须都在语料里；2) 引用的条文号必须真实存在；
  3) 问题不在 jpkb 里（否则答案就是知识库里的同一行，等于自己考自己）；
  4) 按领域均衡抽。

注意：还会额外统计这些问题的「BM25 快通道命中率」——2026-10-14 实测 108 条候选里
102 条（94%）会被快通道截获，只有 6 条能走到 RAG。所以本评测集**不按快通道过滤**，
而是评测时直接调 RAG 通道（评测对象是 RAG 能力，不是快通道门控）。
要复现「只保留确实会走 RAG 的问题」用 --require-rag-path。

用法:
  python tools/build_law_ragas_set.py --n 100
产物: rag_qa/data/law_eval/ragas/law_ragas_set.json
"""
import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for p in (ROOT, TOOLS):
    sys.path.insert(0, p)

import sample_law_faq as sf                                   # noqa: E402

from mysql_qa.db.mysql_client import MySQLClient              # noqa: E402
from mysql_qa.retrieval.bm25_search import BM25Search         # noqa: E402
from mysql_qa.utils.preprocess import preprocess_text         # noqa: E402

OUT_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas", "law_ragas_set.json")
FAST_CHANNEL_THRESHOLD = 0.85


def main():
    ap = argparse.ArgumentParser(description="构建法律版 RAGAS 评测集")
    ap.add_argument("--src", default=sf.SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--require-rag-path", action="store_true",
                    help="只保留不会被 BM25 快通道命中（确实会走 RAG）的问题")
    ap.add_argument("--sample-for-stat", type=int, default=500,
                    help="额外抽多少条来判断池子整体的快通道命中率")
    args = ap.parse_args()

    corpus, numbers, total = sf.load_corpus()
    print(f"语料：{len(corpus)} 部法律 / {total} 条法条", flush=True)

    mysql = MySQLClient()
    in_jpkb = {re.sub(r"\s+", "", q[0]) for q in mysql.fetch_questions()}
    print(f"jpkb 已有问题：{len(in_jpkb)} 条（这些要排除，否则会被快通道截走）", flush=True)

    # 1) 逐行筛选，按领域入库
    pool, seen, stat = {}, set(), {"源行数": 0, "引用语料外法律": 0, "条号不存在": 0,
                                   "已在 jpkb": 0, "重复": 0, "可用": 0}
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stat["源行数"] += 1
            o = json.loads(line)
            q = sf.clean_labels(o.get("question", ""), corpus)
            a = sf.clean_labels(o.get("answer", ""), corpus)
            labels = sf.BOOK_RE.findall(a)
            if not labels:
                continue
            parsed = [sf.parse_cited(x, corpus) for x in labels]
            if any(law is None for law, _ in parsed):
                stat["引用语料外法律"] += 1
                continue
            bad = False
            for law, num in sf.CITE_RE.findall(a):
                law_n, _h = sf.parse_cited(law, corpus)
                v = sf.cn2int(num)
                if law_n in numbers and (v is None or v not in numbers[law_n]):
                    bad = True
                    break
            if bad:
                stat["条号不存在"] += 1
                continue
            key = re.sub(r"\s+", "", q)
            if key in in_jpkb:
                stat["已在 jpkb"] += 1
                continue
            if key in seen:
                stat["重复"] += 1
                continue
            seen.add(key)
            cited = {law for law, _ in parsed}
            hints = {h for _, h in parsed if h}
            d = sf.domain_of(cited, hints, q + a)
            pool.setdefault(d, []).append({"question": q, "ground_truth": a, "domain": d,
                                           "cited_laws": sorted(cited)})
            stat["可用"] += 1

    print("筛选统计：" + json.dumps(stat, ensure_ascii=False), flush=True)
    print("各领域可用量：" + json.dumps({d: len(v) for d, v in pool.items()}, ensure_ascii=False), flush=True)

    # 2) 按领域均衡抽评测集
    rnd = random.Random(args.seed)
    quota = max(1, args.n // len(sf.DOMAIN_ORDER))
    by_domain = {}
    all_items = []
    for d in sf.DOMAIN_ORDER:
        items = list(pool.get(d, []))
        rnd.shuffle(items)
        by_domain[d] = items
        all_items.extend(items)
    chosen, idx = [], 0
    while len(chosen) < args.n:
        progressed = False
        for d in sf.DOMAIN_ORDER:
            items = by_domain[d]
            if idx < len(items) and len(chosen) < args.n:
                chosen.append(items[idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    print(f"抽到 {len(chosen)} 条（目标 {args.n} 条，领域配额 {quota}/域）", flush=True)

    # 3) 算 BM25 快通道命中率（评测集 + 池子随机样本）
    bm25 = BM25Search(None, mysql)
    stat_sample = rnd.sample(all_items, min(args.sample_for_stat, len(all_items)))
    print(f"统计池子整体命中率：法条语料 {len(bm25.article_meta or [])} 条 / FAQ {len(bm25.faq_questions or [])} 条",
          flush=True)

    def score_of(item):
        tokens = preprocess_text(item["question"])
        _, a_score = bm25._best(bm25.article_bm25, tokens)
        _, f_score = bm25._best(bm25.faq_bm25, tokens)
        return round(a_score, 4), round(f_score, 4)

    hit = sum(1 for it in stat_sample if max(score_of(it)) > FAST_CHANNEL_THRESHOLD)
    print(f"池子随机 {len(stat_sample)} 条：会被快通道命中 {hit} 条 "
          f"（{hit / len(stat_sample):.1%}，即线上不会走到 RAG）", flush=True)

    for it in chosen:
        it["article_score"], it["faq_score"] = score_of(it)
    if args.require_rag_path:
        before = len(chosen)
        chosen = [it for it in chosen if max(it["article_score"], it["faq_score"]) <= FAST_CHANNEL_THRESHOLD]
        print(f"[--require-rag-path] {before} 条 -> {len(chosen)} 条", flush=True)
    hit_set = sum(1 for it in chosen if max(it["article_score"], it["faq_score"]) > FAST_CHANNEL_THRESHOLD)
    print(f"评测集里的快通道命中情况：{hit_set}/{len(chosen)} 条会被截获"
          f"（评测时直接走 RAG 通道，不经过快通道）", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=1)
    dom_stat = {}
    for it in chosen:
        dom_stat[it["domain"]] = dom_stat.get(it["domain"], 0) + 1
    print(f"评测集：{len(chosen)} 条 -> {os.path.relpath(args.out, ROOT)}", flush=True)
    print("领域分布：" + json.dumps(dom_stat, ensure_ascii=False), flush=True)
    print("\n样例：")
    for it in chosen[:3]:
        print(f"  [{it['domain']}] {it['question']}\n    参考答案：{it['ground_truth'][:80]}...")


if __name__ == "__main__":
    main()
