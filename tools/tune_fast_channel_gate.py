# -*- coding: utf-8 -*-
"""
快通道门控标定（P2 第三步）
==========================
问题：BM25 快通道用「全库 softmax 分 > 0.85」判定命中，语料扩到 2 万级后这个尺度过松，
      真实法律问题 90% 被抢答、其中约一半答错，RAG 基本走不到。

两组带标签数据（都在本机可复现）：
  P 正样本 200 条：law_faq 库里的原始问题（一模一样时**应该**秒回）
  N 负样本 100 条：真实法律问题（不在 law_faq 内；**只有答对才允许抢答**，否则应放行给 RAG）

输出：各判据下的「错误抢答 / 放行率 / 秒回率」对照表，用来选阈值。

用法:
  python tools/tune_fast_channel_gate.py --dump      # 只落盘逐条统计
  python tools/tune_fast_channel_gate.py             # 落盘 + 打印判据对照表
"""
import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                          # noqa: E402

from base.law_parser import norm_law_name, cn2int           # noqa: E402
from mysql_qa.db.mysql_client import MySQLClient            # noqa: E402
from mysql_qa.retrieval.bm25_search import BM25Search       # noqa: E402
from mysql_qa.utils.preprocess import preprocess_text       # noqa: E402
from rag_qa.core.citation_check import extract_citations    # noqa: E402

EVAL_SET = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas", "law_ragas_set.json")
STAT_FILE = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas", "gate_stats.json")
NUM_RE = re.compile(r"第(.+?)条")
# 语料里的法律名（去掉「中华人民共和国」前缀），用于判断查询里有没有点名某部法律
LAW_NAMES = set()


def key_of(law, article_no):
    m = NUM_RE.search(article_no or "")
    return (norm_law_name(law), cn2int(m.group(1)) if m else None)


def top2(scores):
    """返回 (top1 下标, top1 值, top2 值)。"""
    order = np.argsort(-scores)
    return int(order[0]), float(scores[order[0]]), float(scores[order[1]])


def stats_for(bm25, question, answer_citations=None):
    tokens = preprocess_text(question)
    tok_set = set(tokens)
    art = bm25.article_bm25.get_scores(tokens)
    faq = bm25.faq_bm25.get_scores(tokens)
    ai, a1, a2 = top2(art)
    fi, f1, f2 = top2(faq)
    law, no, _num, _txt = bm25.article_meta[ai]
    art_key = key_of(law, no)
    faq_q = bm25.faq_questions[fi][0]
    # 结构化判据用的特征：
    #   art_cover   = 查询词有多少比例出现在命中的条文正文里（"查询几乎就是这条文"）
    #   faq_jaccard = 查询词与命中问题的词集合 Jaccard（"就是同一个问题"）
    art_tokens = set(preprocess_text(bm25.article_meta[ai][3]))
    faq_tokens = set(preprocess_text(faq_q))
    art_cover = len(tok_set & art_tokens) / max(1, len(tok_set))
    faq_jac = len(tok_set & faq_tokens) / max(1, len(tok_set | faq_tokens))
    art_soft = float(bm25._softmax(art)[ai])
    faq_soft = float(bm25._softmax(faq)[fi])
    rec = {
        "question": question,
        "article_raw": round(a1, 3), "article_raw2": round(a2, 3), "article_margin": round(a1 - a2, 3),
        "article_softmax": round(art_soft, 4), "article_top1": [law, no, art_key[1]],
        "article_top1_correct": bool(answer_citations is not None and art_key in answer_citations),
        "art_cover": round(art_cover, 3),
        "faq_raw": round(f1, 3), "faq_margin": round(f1 - f2, 3), "faq_softmax": round(faq_soft, 4),
        "faq_top1": faq_q, "faq_exact": re.sub(r"\s+", "", faq_q) == re.sub(r"\s+", "", question),
        "faq_jaccard": round(faq_jac, 3),
        "has_law_name": any(len(n) >= 2 and n in question for n in LAW_NAMES),
    }
    if answer_citations is not None:
        # 标准答案里有几条被引用的法条（用于统计 N 的正确答案）
        rec["want_n"] = len(answer_citations)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="只落盘统计，不打印对照表")
    ap.add_argument("--p-n", type=int, default=200, help="正样本条数")
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    mysql = MySQLClient()
    bm25 = BM25Search(None, mysql)
    global LAW_NAMES
    LAW_NAMES = {norm_law_name(law) for law, _no, _num, _txt in bm25.article_meta}
    print(f"语料法律名 {len(LAW_NAMES)} 个", flush=True)
    rnd = random.Random(args.seed)

    # P：库里原句（期望秒回）
    questions = [q[0] for q in bm25.faq_questions]
    pos = rnd.sample(questions, args.p_n)
    P = [stats_for(bm25, q) for q in pos]

    # N：真实法律问题（不在库里；只有答对才允许抢答）
    items = json.load(open(EVAL_SET, encoding="utf-8"))
    N = []
    faq_answer_hit = {}
    for it in items:
        want = extract_citations(it["ground_truth"])
        rec = stats_for(bm25, it["question"], answer_citations=want)
        ans = mysql.fetch_answer(rec["faq_top1"]) or ""
        faq_answer_hit[rec["question"]] = bool(extract_citations(ans) & want)
        rec["faq_top1_answer_ok"] = faq_answer_hit[rec["question"]]
        rec["domain"] = it.get("domain")
        N.append(rec)

    os.makedirs(os.path.dirname(STAT_FILE), exist_ok=True)
    json.dump({"P": P, "N": N}, open(STAT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"逐条统计已落盘：{os.path.relpath(STAT_FILE, ROOT)}（P={len(P)} N={len(N)}）", flush=True)
    if args.dump:
        return

    def evaluate(rule_art, rule_faq, name):
        """返回 (错误抢答数(N), 正确抢答数(N), 放行数(N), P 秒回数, P 被抢答数)。"""
        n_bad = n_ok = n_pass = 0
        for r in N:
            art_fire = rule_art(r)
            faq_fire = rule_faq(r)
            if art_fire or faq_fire:
                correct = (r["article_top1_correct"] if art_fire else r["faq_top1_answer_ok"])
                if correct:
                    n_ok += 1
                else:
                    n_bad += 1
            else:
                n_pass += 1
        p_hit = sum(1 for r in P if rule_art(r) or rule_faq(r))
        print(f"{name:38s} N: 错答 {n_bad:3d}  答对 {n_ok:3d}  放行 {n_pass:3d} | 放行率 {n_pass / len(N):.0%}"
              f" | P 秒回 {p_hit:3d}/{len(P)} = {p_hit / len(P):.0%}", flush=True)
        return dict(name=name, n_bad=n_bad, n_ok=n_ok, n_pass=n_pass,
                    pass_rate=round(n_pass / len(N), 3), p_hit=round(p_hit / len(P), 3))

    print("\n== 判据对照（N=真实法律问题 100 条，P=库里原句 200 条）==")
    rules = []
    # 现状
    rules.append(evaluate(lambda r: r["article_softmax"] > .85,
                          lambda r: r["faq_softmax"] > .85, "现状：softmax > 0.85"))
    for t in (0.9, 0.95, 0.99):
        rules.append(evaluate(lambda r, t=t: r["article_softmax"] > t,
                              lambda r, t=t: r["faq_softmax"] > t, f"softmax > {t}"))
    # 原始分 + 分差
    for a_raw, f_raw, margin in ((20, 20, 3), (25, 25, 5), (30, 30, 5), (25, 25, 0)):
        rules.append(evaluate(lambda r, a=a_raw, m=margin: r["article_raw"] >= a and r["article_margin"] >= m,
                              lambda r, f=f_raw, m=margin: r["faq_raw"] >= f and r["faq_margin"] >= m,
                              f"raw≥{a_raw}/{f_raw} 且分差≥{margin}"))
    # FAQ 只认「同一个问题」，法条用原始分
    for a_raw in (20, 25, 30):
        rules.append(evaluate(lambda r, a=a_raw: r["article_raw"] >= a,
                              lambda r: r["faq_exact"], f"法条 raw≥{a_raw} / FAQ 仅同一问题"))
    # 两个都只在高置信时才抢：FAQ 必须同一问题或极高分
    rules.append(evaluate(lambda r: r["article_raw"] >= 25,
                          lambda r: r["faq_exact"] or r["faq_raw"] >= 30, "法条 raw≥25 / FAQ 同题或 raw≥30"))
    # 结构化判据：FAQ 只认「同一个问题」，法条只认「查询基本被条文覆盖」
    rules.append(evaluate(lambda r: r["art_cover"] >= 0.85,
                          lambda r: r["faq_jaccard"] >= 0.9, "结构：法条覆盖率≥0.85 / FAQ Jaccard≥0.9"))
    rules.append(evaluate(lambda r: r["art_cover"] >= 0.85 and r["article_raw"] >= 15,
                          lambda r: r["faq_jaccard"] >= 0.9, "结构+raw：法条覆盖率≥0.85且raw≥15 / FAQ≥0.9"))
    rules.append(evaluate(lambda r: r["has_law_name"] and r["art_cover"] >= 0.8,
                          lambda r: r["faq_jaccard"] >= 0.9, "结构：法条需含法名且覆盖率≥0.8 / FAQ≥0.9"))
    rules.append(evaluate(lambda r: r["art_cover"] >= 0.9,
                          lambda r: r["faq_jaccard"] >= 0.95, "结构严格：法条≥0.9 / FAQ≥0.95"))
    json.dump(rules, open(os.path.join(os.path.dirname(STAT_FILE), "gate_rules.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
