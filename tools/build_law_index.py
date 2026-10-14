# -*- coding: utf-8 -*-
"""
把 law_data 切分并灌入 Milvus 集合（设计文档§7 步骤 3~4）
流程: process_documents(law_data) -> 法条边界切分 -> BGE-M3 向量化 -> 写入集合 lawrag。
分批写入，便于观察进度、失败可重跑（按内容 md5 幂等 upsert）。
用法:
  python tools/build_law_index.py
  python tools/build_law_index.py --batch 256
  python tools/build_law_index.py --reset    # 先删掉旧集合再灌（语料换过之后必须用）
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.config import config                                # noqa: E402
from rag_qa.core.document_processor import process_documents  # noqa: E402
from rag_qa.core.vector_store import VectorStore              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="法律语料切分并灌入 Milvus")
    ap.add_argument("--batch", type=int, default=256, help="每批写入条数")
    ap.add_argument("--reset", action="store_true",
                    help="先删除集合再重建（语料扩容/更换后必须用，否则新旧语料块混在一个集合里）")
    args = ap.parse_args()

    start = time.time()
    if args.reset:
        from pymilvus import MilvusClient  # 只删集合，不必加载向量与重排模型
        client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
                              db_name=config.MILVUS_DATABASE_NAME)
        name = config.MILVUS_COLLECTION_NAME
        if client.has_collection(name):
            client.drop_collection(name)
            print(f"[0/3] 已删除旧集合 {name}（避免新旧语料块混在一个集合里）")
        else:
            print(f"[0/3] 集合 {name} 不存在，直接新建")
    docs = process_documents(config.DATA_DIR + "/law_data")
    print(f"[1/3] 法条切分完成: {len(docs)} 个 child 块")
    print(f"[2/3] 初始化向量库 (collection={config.MILVUS_COLLECTION_NAME}) ...")
    vector_store = VectorStore()
    print(f"      collection={vector_store.collection_name}, dense_dim={vector_store.dense_dim}")
    print("[3/3] 分批写入 ...")
    done = 0
    for i in range(0, len(docs), args.batch):
        chunk = docs[i:i + args.batch]
        vector_store.add_documents(chunk)
        done += len(chunk)
        print(f"      已灌 {done}/{len(docs)}  ({time.time() - start:.0f}s)")
    print(f"完成，总耗时 {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
