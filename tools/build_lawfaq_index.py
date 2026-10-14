# -*- coding: utf-8 -*-
"""
把 law_faq 的法律问答灌进 Milvus「法律问答」集合（P2 第三召回源）
=============================================================
为什么单独一个集合：法条集合的检索强制 `status == "有效"`（P1 加的），
FAQ 条目没有 status，混进去会被整条过滤掉。

用法:
  python tools/build_lawfaq_index.py            # 增量写入（同 id 覆盖）
  python tools/build_lawfaq_index.py --reset    # 删掉集合重建（推荐第一次这么跑）
  python tools/build_lawfaq_index.py --limit 500
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mysql_qa.db.mysql_client import MySQLClient                      # noqa: E402
from rag_qa.core.vector_store import VectorStore, FAQ_COLLECTION_NAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="构建 law_faq 问答向量集合")
    ap.add_argument("--reset", action="store_true", help="删掉集合重建")
    ap.add_argument("--limit", type=int, default=0, help="只灌前 N 条（调试用）")
    ap.add_argument("--batch", type=int, default=256, help="向量化批大小")
    args = ap.parse_args()

    mysql = MySQLClient()
    mysql.cursor.execute("SELECT domain, question, answer FROM law_faq")
    records = [{"domain": s, "question": q, "answer": a}
               for s, q, a in mysql.cursor.fetchall()]
    if args.limit:
        records = records[:args.limit]
    print(f"law_faq 读出 {len(records)} 条", flush=True)

    t0 = time.time()
    vs = VectorStore()          # 构造时会把空集合准备好
    if args.reset and vs.client.has_collection(FAQ_COLLECTION_NAME):
        vs.client.drop_collection(FAQ_COLLECTION_NAME)
        vs._ensure_faq_collection()
        print(f"已删除并重建空集合 {FAQ_COLLECTION_NAME}", flush=True)

    n = vs.add_faq_documents(records, batch_size=args.batch)
    stats = vs.client.get_collection_stats(FAQ_COLLECTION_NAME)
    print(f"写入 {n} 条，集合 {FAQ_COLLECTION_NAME} 现有 {stats['row_count']} 条，"
          f"用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
