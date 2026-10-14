# -*- coding: utf-8 -*-
"""
法律数据入库脚本
================
把 rag_qa/data/law_data 的结构化结果写进 MySQL（《项目方案书》§7 步骤 1、§4.1）：
  1) law_articles 表  <- law_data 解析出的全部法条（P0 扩容后：287 部法律 / 21255 条法条）
  2) law_faq 表          <- mysql_qa/data/law_qa.csv（由 tools/sample_law_faq.py 生成的 2 万条法律 FAQ）

用法:
  python tools/import_law_data.py            # 入库 law_articles + 重建 law_faq
  python tools/import_law_data.py --law-only # 只入库 law_articles
  python tools/import_law_data.py --qa-only  # 只重建 law_faq

注意:
  - 重建 law_faq 会先清空该表；历史问答数据已归档在 `_archive/`，需要回滚时从那里取。
  - law_qa.csv 由 tools/sample_law_faq.py 生成（领域均衡 + 引用可核对），本脚本只负责入库。
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mysql_qa.db.mysql_client import MySQLClient            # noqa: E402
from base.law_parser import parse_law_dir, load_law_meta, chapter_of   # noqa: E402

LAW_DATA_DIR = os.path.join(ROOT, "rag_qa", "data", "law_data")
LAW_QA_CSV = os.path.join(ROOT, "mysql_qa", "data", "law_qa.csv")
LAW_META = os.path.join(ROOT, "rag_qa", "data", "law_meta.json")


def import_law_articles(client):
    # 每行带上 P1 元数据：编/章/节、公布日、施行日、时效状态（历史版本 status 为 已修改/已废止）
    meta = load_law_meta(LAW_META)
    rows = []
    for law_name, article_no, article_num, article_text in parse_law_dir(LAW_DATA_DIR):
        info = meta.get(law_name, {})
        rows.append((law_name, article_no, article_num, article_text,
                     chapter_of(meta, law_name, article_num),
                     info.get("publish_date", ""), info.get("effective_date", ""),
                     info.get("status", "有效")))
    client.create_law_articles_table()
    # 重建语义：先清空再写。否则 insert_law_articles 用的 INSERT IGNORE 会把
    # 和旧语料重号的法条（如民法典每条）静默跳过，库里的条数就永远对不上。
    client.cursor.execute("TRUNCATE TABLE law_articles")
    client.insert_law_articles(rows)
    client.cursor.execute("SELECT COUNT(*) FROM law_articles")
    return len(rows), client.cursor.fetchone()[0]


def import_law_qa(client):
    if not os.path.exists(LAW_QA_CSV):
        raise SystemExit(f"缺少 {LAW_QA_CSV}，请先运行: python tools/sample_law_faq.py")
    with open(LAW_QA_CSV, encoding="utf-8") as fi:
        n = sum(1 for _ in csv.DictReader(fi))
    client.cursor.execute("TRUNCATE TABLE law_faq")
    client.insert_faq_csv(LAW_QA_CSV)
    client.cursor.execute("SELECT COUNT(*) FROM law_faq")
    return n, client.cursor.fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description="法律数据入库")
    ap.add_argument("--law-only", action="store_true", help="只入库 law_articles")
    ap.add_argument("--qa-only", action="store_true", help="只重建 law_faq 法律问答")
    args = ap.parse_args()

    client = MySQLClient()
    try:
        if not args.qa_only:
            parsed, total = import_law_articles(client)
            client.cursor.execute("SELECT status, COUNT(*) FROM law_articles GROUP BY status")
            stat = dict(client.cursor.fetchall())
            print(f"[law_articles] 解析 {parsed} 条 -> 库内 {total} 条，按时效分布 {stat}")
        if not args.law_only:
            built, total = import_law_qa(client)
            print(f"[law_faq] 生成 {built} 条法律问答 -> 库内 {total} 条 (CSV: {LAW_QA_CSV})")
    finally:
        client.close()


if __name__ == "__main__":
    main()
