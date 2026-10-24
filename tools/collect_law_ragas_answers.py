# -*- coding: utf-8 -*-
"""
跑 RAG 通道收集答案与上下文，产出 RAGAS 输入（P2 第二步）
=========================================================
对 build_law_ragas_set.py 建好的评测集逐条执行**线上 RAG 通道**：
    BERT 意图识别 → 策略选择 → 检索(混合+精排) → 拼上下文 → LLM 流式生成
故意不经过快通道（条文号路由 / BM25）——因为线上 86.6% 的同类问题会被快通道截住，
那是「门控」问题；这里要量的是 RAG 本身的检索与生成质量。

LLM 调用（策略选择、生成）走线程并发，GPU 侧（意图分类、向量检索）加锁串行，
避免显存/线程安全问题。

用法:
  python tools/collect_law_ragas_answers.py --workers 8
产物: rag_qa/data/law_eval/ragas/law_ragas_answers.json
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.config import config                              # noqa: E402
from new_main import IntegratedQASystem                     # noqa: E402

SET_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas", "law_ragas_set.json")
OUT_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "law_eval", "ragas", "law_ragas_answers.json")

gpu_lock = threading.Lock()   # 意图分类 / 向量检索 / 精排 都吃 GPU，串行执行


def run_one(qa, item, done, total, t0, domain=None, domain_mode="off", allow_faq=True):
    """返回一条 RAGAS 记录；失败时带上 error 字段。"""
    q = item["question"]
    try:
        with gpu_lock:
            category = qa.rag_system.query_classifier.predict_category(q)
        strategy = qa.rag_system.strategy_selector.select_strategy(q)
        with gpu_lock:
            docs = qa.rag_system.retrieve_and_merge(
                q, source_filter="law", strategy=strategy,
                domain=domain, domain_mode=domain_mode, allow_faq=allow_faq)
            contexts = [qa.rag_system._format_context_doc(d) for d in docs] if docs else []
            prompt = qa.rag_system.rag_prompt.format(
                context="\n\n".join(contexts), history="", question=q,
                phone=config.CUSTOMER_SERVICE_PHONE)
        answer = "".join(qa.rag_system.llm(prompt))
        with done["lock"]:
            done["n"] += 1
            print(f"  {done['n']}/{total} [{item.get('domain')}] {q[:26]}... "
                  f"strategy={strategy} 上下文 {len(contexts)} 块 ({time.time() - t0:.0f}s)", flush=True)
        return {"question": q, "contexts": contexts, "answer": answer,
                "ground_truth": item["ground_truth"], "domain": item.get("domain"),
                "strategy": strategy, "category": category,
                "fast_channel_score": max(item.get("article_score", 0), item.get("faq_score", 0))}
    except Exception as e:  # 单条失败不拖垮整轮
        with done["lock"]:
            done["n"] += 1
            print(f"  {done['n']}/{total} 失败：{q[:30]}... {type(e).__name__}: {e}", flush=True)
        return {"question": q, "contexts": [], "answer": "", "ground_truth": item["ground_truth"],
                "domain": item.get("domain"), "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description="跑 RAG 通道收集答案/上下文 -> RAGAS 输入")
    ap.add_argument("--data", default=SET_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用，0=全部）")
    ap.add_argument("--resume", action="store_true",
                    help="已有产物时只补跑「答案为空」的条目（额度中断后接着跑）")
    ap.add_argument("--domain", default=None, help="领域视图名（如 labor）；默认不过滤")
    ap.add_argument("--domain-mode", default="off", choices=["off", "soft", "strict"])
    ap.add_argument("--no-faq", action="store_true", help="不注入 FAQ 第三召回源")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]

    prev = {}
    if args.resume and os.path.exists(args.out):
        for r in json.load(open(args.out, encoding="utf-8")):
            if r.get("answer"):
                prev[r["question"]] = r
        print(f"[--resume] 已有 {len(prev)} 条答案，只补跑剩下的", flush=True)

    todo = [(i, it) for i, it in enumerate(items) if it["question"] not in prev]
    print(f"评测集：{len(items)} 条，待跑 {len(todo)} 条"
          f"（{os.path.relpath(args.data, ROOT)}），并发 {args.workers} | 模型 {config.LLM_MODEL}"
          f" | 领域 {args.domain or 'off'}/{args.domain_mode} | FAQ {not args.no_faq}", flush=True)

    t0 = time.time()
    qa = IntegratedQASystem() if todo else None
    print(f"[init] {time.time() - t0:.1f}s", flush=True)

    done = {"n": 0, "lock": threading.Lock()}
    results = [None] * len(items)
    for i, it in enumerate(items):
        if it["question"] in prev:
            results[i] = prev[it["question"]]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, qa, it, done, len(todo), t0,
                             args.domain, args.domain_mode, not args.no_faq): i
                   for i, it in todo}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    ok = [r for r in results if r.get("answer")]
    empty_ctx = [r for r in ok if not r["contexts"]]
    avg_ctx = sum(len(r["contexts"]) for r in ok) / max(1, len(ok))
    print(f"\n完成：成功 {len(ok)}/{len(results)}，耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    print(f"平均上下文块数 {avg_ctx:.1f}，检索为空 {len(empty_ctx)} 条", flush=True)
    print(f"策略分布：" + json.dumps(
        {s: sum(1 for r in ok if r.get("strategy") == s)
         for s in {r.get("strategy") for r in ok}}, ensure_ascii=False), flush=True)
    print(f"产物：{os.path.relpath(args.out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
