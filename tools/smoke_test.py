# -*- coding: utf-8 -*-
"""
项目自检（改名 / 重建索引 / 换模型之后跑一次）
==============================================
只做**小规模**验证，不批量跑评测、不消耗大量模型额度：
  1) 三个外部服务连通：MySQL / Redis / Milvus
  2) 语料规模：law_articles / law_faq / lawrag / lawfaq
  3) BM25 快通道：库内原句应秒回（词面门控），真实问题应放行给 RAG
  4) 条文号精确路由：《X法》第 Y 条 直接查 MySQL
  5) RAG 通道端到端一条（会调用一次大模型；加 --no-llm 可跳过）

用法:
  python tools/smoke_test.py
  python tools/smoke_test.py --no-llm
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.config import config                                  # noqa: E402

OK, BAD = "  [OK]  ", "  [FAIL]"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"{OK if ok else BAD} {name}{(' | ' + detail) if detail else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="项目自检")
    ap.add_argument("--no-llm", action="store_true", help="跳过 RAG 端到端（不调用大模型）")
    args = ap.parse_args()

    t0 = time.time()
    print(f"模型: {config.LLM_MODEL} | 集合: {config.MILVUS_COLLECTION_NAME} | "
          f"来源: {config.VALID_SOURCES}\n")

    # ---------- 1) 外部服务 ----------
    from mysql_qa.db.mysql_client import MySQLClient
    from mysql_qa.cache.redis_client import RedisClient
    mysql = MySQLClient()
    redis = RedisClient()
    check("MySQL 连通", mysql.cursor is not None)
    try:
        redis.client.ping()
        check("Redis 连通", True)
    except Exception as e:
        check("Redis 连通", False, str(e)[:80])

    # ---------- 2) 语料规模 ----------
    for table in ("law_articles", "law_faq"):
        try:
            mysql.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            n = mysql.cursor.fetchone()[0]
            check(f"MySQL 表 {table}", n > 0, f"{n} 行")
        except Exception as e:
            check(f"MySQL 表 {table}", False, str(e)[:80])

    services_ok = True
    try:
        from pymilvus import MilvusClient
        from rag_qa.core.vector_store import FAQ_COLLECTION_NAME

        mc = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
                          db_name=config.MILVUS_DATABASE_NAME)
        for coll in (config.MILVUS_COLLECTION_NAME, FAQ_COLLECTION_NAME):
            n = mc.get_collection_stats(coll).get("row_count")
            check(f"Milvus 集合 {coll}", bool(n), f"{n} 行")
    except Exception as e:
        services_ok = False
        check("Milvus 集合", False, str(e)[:120])

    # ---------- 3~5) 走真实系统 ----------
    from new_main import IntegratedQASystem
    qa = IntegratedQASystem()
    check("系统初始化（模型 + 检索器）", True, f"{time.time() - t0:.0f}s")

    # 3) 快通道：库内原句（词集合完全一致 -> 应命中）
    mysql.cursor.execute("SELECT question FROM law_faq LIMIT 1")
    in_lib = mysql.cursor.fetchone()[0]
    ans, need_rag = qa.bm25_search.search(in_lib)
    check("快通道：库内原句秒回", bool(ans) and not need_rag, in_lib[:24] + "…")

    # 3) 快通道：真实口语问题（不在库里 -> 应放行给 RAG）
    real_q = "公司一直不给我交社保怎么办"
    ans2, need_rag2 = qa.bm25_search.search(real_q)
    check("快通道：真实问题放行给 RAG", ans2 is None and need_rag2, real_q)

    # 4) 条文号精确路由
    citation_ans = qa._answer_by_citation("《中华人民共和国民法典》第一千零六十二条")
    check("条文号精确路由", bool(citation_ans), (citation_ans or "")[:38] + "…")

    # 5) RAG 端到端（一次大模型调用）
    if args.no_llm:
        print("  [跳过] RAG 端到端（--no-llm）")
    else:
        q = "租房到期押金不退怎么办？"
        answer = "".join(tok for tok, _ in qa.query(q))
        has_cite = "《" in answer and "》第" in answer
        check("RAG 端到端（1 次模型调用）", len(answer) > 20, f"{len(answer)} 字")
        check("答案带《法律名》第X条 引用", has_cite)
        print(f"\n  问题：{q}\n  回答：{answer[:120]}…")

    mysql.close()
    bad = [n for n, ok in results if not ok]
    print(f"\n合计 {len(results)} 项，通过 {len(results) - len(bad)} 项"
          f"{'，失败：' + '、'.join(bad) if bad else '，全部通过'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
