# -*- coding: utf-8 -*-
"""BM25 快通道 / BERT 意图识别 体检脚本（只读，不写库、不写 Redis）

用途：回答「意图识别该不该用到 BM25 / 该不该前置」这类问题，给出可复现的量化依据。
三组样本（同一次随机种子）：
  A 库内 FAQ 原句（法律域上限）  B 库外法律问题（泛化）  C 通用/非法律问题（误命中）

用法：
  python tools/probe_bm25_route.py            # 快通道命中率 + 意图分类判对率
  python tools/probe_bm25_route.py --raw      # 追加 raw 分 / 归一化分对照
输出：
  _probe_bm25_route_result.json（工作目录） + 屏幕摘要
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from base.config import config
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25_search import BM25Search
from mysql_qa.utils.preprocess import preprocess_text
from rag_qa.core.query_classifier import QueryClassifier

THRESHOLD = 0.85
SEED = 20261014
FAQ_TEST = "rag_qa/data/law_eval/qa/laws_data_50k_test.jsonl"

GENERIC_QUERIES = [
    "今天北京天气怎么样", "Python 里怎么读取 CSV 文件", "帮我写一首关于春天的诗",
    "PyCharm 专业版怎么激活", "上海到杭州的高铁要多久", "推荐几本学机器学习的书",
    "怎么用 Excel 做数据透视表", "什么是量子纠缠", "我的电脑开机很慢怎么办",
    "怎样做红烧肉", "猫为什么会发出呼噜声", "英语四六级怎么备考",
    "水星是太阳系第几颗行星", "怎么减肥比较健康", "介绍一下 Python 的列表推导式",
    "帮我润色一段自我介绍", "中国有多少个省份", "篮球规则里什么是走步",
    "冰箱冷藏室温度设置多少合适", "怎么给孩子选幼儿园", "微信怎么设置消息免打扰",
    "求推荐一首适合跑步听的歌", "咖啡一天喝多少合适", "怎么把照片的背景换成白色",
    "3 的 5 次方等于多少",
]

RAW_QUERIES = [
    "今天北京天气怎么样", "3 的 5 次方等于多少", "介绍一下 Python 的列表推导式",
    "怎样做红烧肉", "冰箱冷藏室温度设置多少合适", "公司一直不给我交社保怎么办",
    "劳动合同解除后经济补偿怎么算",
]


def quantile(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))], 4)


def build_sample_sets(bm25_faq_questions, all_law_questions):
    """同一随机种子下复现 A/B/C 三组，保证与文档里的数字对得上。"""
    random.seed(SEED)
    in_corpus = [q[0] for q in bm25_faq_questions]
    in_set = set(in_corpus)
    out_corpus = [q for q in all_law_questions if q not in in_set]
    return [
        ("A 库内原句(法律域上限)", random.sample(in_corpus, 100), "法律咨询"),
        ("B 库外法律问题(泛化)", random.sample(out_corpus, 100), "法律咨询"),
        ("C 通用/非法律问题(误命中)", GENERIC_QUERIES, "通用知识"),
    ]


def probe_bm25(bm25, sets):
    out = []
    for name, queries, _expect in sets:
        art, faq, rows = [], [], []
        for q in queries:
            tokens = preprocess_text(q)
            ai, ascore = bm25._best(bm25.article_bm25, tokens)
            fi, fscore = bm25._best(bm25.faq_bm25, tokens)
            art.append(ascore)
            faq.append(fscore)
            rows.append({
                "query": q,
                "article_score": round(ascore, 4),
                "article_top1": (f"{bm25.article_meta[ai][0]} {bm25.article_meta[ai][1]}"
                                 if ai >= 0 else None),
                "faq_score": round(fscore, 4),
                "faq_top1": bm25.faq_questions[fi][0] if fi >= 0 else None,
            })
        n = len(queries)
        out.append({
            "set": name,
            "n": n,
            "article_hit_rate@0.85": round(sum(s > THRESHOLD for s in art) / n, 3),
            "faq_hit_rate@0.85": round(sum(s > THRESHOLD for s in faq) / n, 3),
            "article_score": {"p50": quantile(art, .5), "p90": quantile(art, .9), "max": round(max(art), 4)},
            "faq_score": {"p50": quantile(faq, .5), "p90": quantile(faq, .9), "max": round(max(faq), 4)},
            "rows": rows,
        })
    return out


def probe_raw(bm25):
    print("\n-- raw BM25 vs softmax 归一化分 --", flush=True)
    for q in RAW_QUERIES:
        tokens = preprocess_text(q)
        for label, model in (("法条", bm25.article_bm25), ("FAQ ", bm25.faq_bm25)):
            if model is None:
                continue
            scores = model.get_scores(tokens)
            top = np.argsort(-scores)[:1]
            sm = bm25._softmax(scores)
            print(f"{q} | {label} | raw_top1={float(scores[top[0]]):.2f}"
                  f" | softmax_top1={float(sm[top[0]]):.4f}"
                  f" | 非零文档={int((scores > 0).sum())}/{len(scores)}", flush=True)


def probe_intent(sets):
    clf = QueryClassifier(model_path=f"{config.MODELS_DIR}/bert_query_classifier_law")
    out = []
    print("", flush=True)
    for name, queries, expect in sets:
        wrong = [[q, clf.predict_category(q)] for q in queries]
        wrong = [w for w in wrong if w[1] != expect]
        n = len(queries)
        print(f"[意图] {name} n={n} 判对率={(n - len(wrong)) / n:.3f} 期望={expect}", flush=True)
        for q, got in wrong[:10]:
            print(f"    误判: {q} -> {got}", flush=True)
        out.append({"set": name, "n": n, "expect": expect, "wrong": wrong})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="追加 raw 分对照")
    parser.add_argument("--no-intent", action="store_true", help="跳过 BERT 意图分类")
    args = parser.parse_args()

    t0 = time.time()
    mysql = MySQLClient()
    with open(FAQ_TEST, encoding="utf-8") as f:
        law_questions = [json.loads(line)["question"] for line in f if line.strip()]

    bm25 = BM25Search(None, mysql)  # 只做打分，无需 Redis
    sets = build_sample_sets(bm25.faq_questions, law_questions)
    result = {"route": probe_bm25(bm25, sets)}
    if args.raw:
        probe_raw(bm25)
    for r in result["route"]:
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, ensure_ascii=False), flush=True)

    if not args.no_intent:
        result["intent"] = probe_intent(sets)

    with open("_probe_bm25_route_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n[done] {time.time() - t0:.1f}s -> _probe_bm25_route_result.json", flush=True)


if __name__ == "__main__":
    main()
