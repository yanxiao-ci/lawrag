# -*- coding: utf-8 -*-
"""
语料正文完整性检查
==================
现有四类评测（Recall / 覆盖率 / 门控 / RAGAS）都只看条号，不看正文。
本脚本直接对比「原始 parquet 的法条正文」与「law_data/*.txt / MySQL law_articles」，
量化枚举项、说明段等续行有没有丢。

用法:
  python tools/check_corpus_integrity.py                 # 只查 .txt
  python tools/check_corpus_integrity.py --mysql         # 同时查 MySQL
  python tools/check_corpus_integrity.py --min-coverage 0.99
  python tools/check_corpus_integrity.py --json-out rag_qa/data/corpus_integrity.json
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from base.law_parser import parse_law_file  # noqa: E402
from expand_law_corpus import (  # noqa: E402
    DEFAULT_SRC_DIR, HISTORY_LAWS_DEFAULT, HISTORY_STATUS, find_src,
    safe_name, select_rows, split_articles,
)

LAW_DATA_DIR = os.path.join(ROOT, "rag_qa", "data", "law_data")
_WS = re.compile(r"\s+")


def _norm(text):
    """只比正文文字，忽略空白差异。"""
    return _WS.sub("", text or "")


def _split_no_body(article_line):
    """'第X条　正文' -> ('第X条', '正文')。"""
    if "　" in article_line:
        no, body = article_line.split("　", 1)
        return no.strip(), body.strip()
    m = re.match(r"^(第[^\s]+条(?:之[^\s]+)?)\s*(.*)$", article_line)
    return (m.group(1), m.group(2).strip()) if m else (article_line.strip(), "")


def expected_by_law(src):
    """按 expand_law_corpus.py 的同一口径，从原始 parquet 生成 {法律名: {条号: 正文}}。"""
    df = pd.read_parquet(src)
    rows, _ = select_rows(df, HISTORY_LAWS_DEFAULT)
    out = {}
    for _, r in rows.iterrows():
        title = safe_name(r["title"])
        if r["type"] == "宪法":
            title = "中华人民共和国宪法"
        if r["status"] in HISTORY_STATUS:
            year = str(r["publish_date"])[:4]
            if not year.isdigit():
                year = str(r["effective_date"])[:4]
            title = f"{title}（{year}年版）"
        if "《" in title or "》" in title:
            continue
        articles, _ = split_articles(r["content"])
        if not articles:
            continue
        out[title] = dict(_split_no_body(a) for a in articles)
    return out


def actual_from_txt(law_dir):
    out = {}
    for fn in sorted(os.listdir(law_dir)):
        if not fn.endswith(".txt"):
            continue
        law = os.path.splitext(fn)[0]
        out[law] = {no: text for _, no, _num, text in parse_law_file(os.path.join(law_dir, fn))}
    return out


def actual_from_mysql():
    from mysql_qa.db.mysql_client import MySQLClient  # 延迟导入，未开 --mysql 时不连库
    client = MySQLClient()
    try:
        client.cursor.execute("SELECT law_name, article_no, article_text FROM law_articles")
        out = {}
        for law, no, text in client.cursor.fetchall():
            out.setdefault(law, {})[no] = text
        return out
    finally:
        client.close()


def compare(expected, actual, label):
    total_exp = total_got = 0
    per_law = []
    missing_top = []
    for law, exp in expected.items():
        got = actual.get(law)
        if got is None:
            per_law.append({"law": law, "articles": len(exp), "coverage": 0.0, "note": "missing file/table"})
            continue
        e_chars = g_chars = 0
        for no, body in exp.items():
            eb = _norm(body)
            gb = _norm(got.get(no, ""))
            e_chars += len(eb)
            g_chars += len(gb)
            if len(eb) and len(gb) < len(eb):
                missing_top.append((len(eb) - len(gb), law, no, len(eb), len(gb)))
        cov = (g_chars / e_chars) if e_chars else 1.0
        total_exp += e_chars
        total_got += g_chars
        per_law.append({"law": law, "articles": len(exp), "coverage": round(cov, 4)})
    missing_top.sort(reverse=True)
    overall = (total_got / total_exp) if total_exp else 1.0
    print(f"\n[{label}] 法律 {len(expected)} 部；正文覆盖率 {overall:.2%}（{total_got:,}/{total_exp:,} 字符）")
    bad = [x for x in per_law if x["coverage"] < 0.95]
    if bad:
        print(f"  低于 95% 的法律 {len(bad)} 部：")
        for x in sorted(bad, key=lambda y: y["coverage"])[:20]:
            print(f"    {x['coverage']:.1%}  {x['law']}  ({x['articles']} 条)")
    print("  丢失最多的条文 Top 10：")
    for diff, law, no, elen, glen in missing_top[:10]:
        print(f"    -{diff:>5} 字  {law} {no}  (原文 {elen} / 现库 {glen})")
    return {
        "label": label,
        "laws": len(expected),
        "overall_coverage": round(overall, 4),
        "low_coverage_laws": sorted(bad, key=lambda y: y["coverage"]),
        "top_missing": [
            {"law": law, "article_no": no, "missing_chars": diff,
             "expected_chars": elen, "actual_chars": glen}
            for diff, law, no, elen, glen in missing_top[:20]
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="语料正文完整性检查")
    ap.add_argument("--src", default=None, help="原始 parquet 路径")
    ap.add_argument("--law-dir", default=LAW_DATA_DIR)
    ap.add_argument("--mysql", action="store_true", help="同时检查 MySQL law_articles")
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    src = find_src(args.src)
    print(f"原始数据源: {src}")
    expected = expected_by_law(src)
    print(f"原始语料解析: {len(expected)} 部 / {sum(len(v) for v in expected.values()):,} 条")

    results = [compare(expected, actual_from_txt(args.law_dir), "law_data/*.txt")]
    if args.mysql:
        try:
            results.append(compare(expected, actual_from_mysql(), "MySQL law_articles"))
        except Exception as e:
            print(f"\n[MySQL] 跳过：{type(e).__name__}: {e}")
            results.append({"label": "MySQL law_articles", "error": str(e)})

    ok = all(r.get("overall_coverage", 0) >= args.min_coverage for r in results if "overall_coverage" in r)
    print(f"\n结论：{'PASS' if ok else 'FAIL'}（阈值 {args.min_coverage:.0%}）")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"src": src, "threshold": args.min_coverage, "results": results},
                      f, ensure_ascii=False, indent=1)
        print(f"已写入 {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
