# -*- coding: utf-8 -*-
"""
百炼模型可用性探测（换模型用）
==============================
百炼上的模型额度是按模型分别计的：某个模型返回 `403 Free quota exhausted` 时，
换一个**较新的、同能力**的模型即可，不需要改生产配置。

本脚本做两件事：
  1) 拉取当前 endpoint 的模型列表（GET /models）；
  2) 逐个发一条最小聊天请求，报告 可用/403/超时 与延迟，按上线时间倒序输出。

用法:
  python tools/probe_llm_models.py                 # 探测聊天类模型
  python tools/probe_llm_models.py --all           # 连 embedding/rerank/图像/语音一起列
  python tools/probe_llm_models.py --models qwen3.6-flash kimi-k3
  python tools/probe_llm_models.py --json out.json

换模型示例（生成/判分都读环境变量，无需改 .env）:
  LLM_MODEL=qwen3.6-flash python tools/collect_law_ragas_answers.py ...
  python rag_qa/rag_assessment/ragas_evaluate.py --judge-model qwen3.6-flash ...
"""
import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from openai import OpenAI  # noqa: E402

from base.config import config  # noqa: E402

# 非对话模型的关键词：默认过滤掉，--all 时保留
NON_CHAT = ("embedding", "rerank", "image", "tts", "asr", "audio", "speech",
            "realtime", "livetranslate", "s2s", "omni", "ocr", "vl-", "mt-",
            "deep-research", "deep-search", "character", "gui-")


def listed(client):
    """返回 [{id, created, date}]，按 created 倒序。"""
    data = client.models.list().data
    rows = [{"id": m.id, "created": m.created} for m in data]
    for r in rows:
        r["date"] = datetime.datetime.fromtimestamp(r["created"]).strftime("%Y-%m-%d")
    return sorted(rows, key=lambda x: -x["created"])


def is_chat(model_id):
    low = model_id.lower()
    return not any(k in low for k in NON_CHAT)


def _chat(client, model_id, extra):
    return client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "只回答两个字：可用"}],
        temperature=0.0, max_tokens=16, **extra)


def probe(client, model_id, timeout):
    """发一条最小请求；返回 (状态, 延迟, 说明)。"""
    t0 = time.time()
    try:
        try:
            # 大多数 Qwen/GLM 支持关闭思考；不支持的模型会报参数错误，下面回退重试
            r = _chat(client, model_id, {"extra_body": {"enable_thinking": False}})
        except Exception as e:
            msg = str(e).lower()
            if "enable_thinking" in msg or "extra_body" in msg or "invalid parameter" in msg:
                r = _chat(client, model_id, {})
            else:
                raise
        text = (r.choices[0].message.content or "").strip()
        return "OK", time.time() - t0, text[:20] or "(空)"
    except Exception as e:
        msg = str(e).replace("\n", " ")
        status = "QUOTA" if "quota" in msg.lower() else "ERR"
        return status, time.time() - t0, msg[:120]


def main():
    ap = argparse.ArgumentParser(description="探测百炼模型可用性")
    ap.add_argument("--models", nargs="*", default=None, help="只探测指定模型")
    ap.add_argument("--all", action="store_true", help="不过滤非对话模型")
    ap.add_argument("--workers", type=int, default=8, help="并发探测数（默认 8）")
    ap.add_argument("--timeout", type=int, default=60, help="单次请求超时秒数")
    ap.add_argument("--json", default=None, help="把结果写到 JSON")
    args = ap.parse_args()

    client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                    base_url=config.DASHSCOPE_BASE_URL,
                    timeout=args.timeout,
                    max_retries=0)
    print(f"endpoint: {config.DASHSCOPE_BASE_URL}", flush=True)

    if args.models:
        rows = [{"id": m, "date": "-", "created": 0} for m in args.models]
    else:
        rows = listed(client)
        if not args.all:
            rows = [r for r in rows if is_chat(r["id"])]
        print(f"模型 {len(rows)} 个（按上线时间倒序）", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda r: probe(client, r["id"], args.timeout), rows))

    for r, (status, dt, note) in zip(rows, results):
        print(f"{r['date']}  {status:5s} {dt:6.1f}s  {r['id']:45s} {note}", flush=True)

    ok = [r["id"] for r, (s, _dt, _n) in zip(rows, results) if s == "OK"]
    print(f"\n可用 {len(ok)}/{len(rows)}：{', '.join(ok) if ok else '(无)'}", flush=True)
    if args.json:
        out = [dict(r, status=s, latency=round(dt, 2), note=n)
               for r, (s, dt, n) in zip(rows, results)]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"结果已写入：{args.json}")


if __name__ == "__main__":
    main()
