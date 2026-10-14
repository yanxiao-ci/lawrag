#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
法律领域二分类训练数据生成器
============================
为改造后的查询分类器（QueryClassifier）生成「通用知识 / 法律咨询」二分类训练集，
沿用「均衡生成」的做法（提示词见 rag_qa/classify_data/提示词模块_法律.txt）。

产物:
  rag_qa/classify_data/model_law_5000.json —— JSONL，一行一条 {"query": "...", "label": "..."}
  （默认 2500「通用知识」+ 2500「法律咨询」= 5000 条，与原有 model_generic_5000.json 同格式）

训练用法（无需改代码，传参即可）:
  classifier.train_model(data_file='../classify_data/model_law_5000.json')

用法:
  python tools/gen_law_classify_data.py
  python tools/gen_law_classify_data.py --per-class 2500 --batch 60 --workers 4
  python tools/gen_law_classify_data.py --rebuild   # 删掉已有产物重新生成

特性:
  - 断点续跑：产物文件里已有的样本会被加载并以去重方式续写。
  - 均衡：按类别分别补齐到目标条数，label 严格控制在两类之内。
  - 去重：按去除空白后的 query 去重。
"""

import argparse
import configparser
import json
import math
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFY_DIR = os.path.join(ROOT, "rag_qa", "classify_data")
OUT_FILE = os.path.join(CLASSIFY_DIR, "model_law_5000.json")

LABELS = ["通用知识", "法律咨询"]

GENERAL_DESC = ("内容为数学计算、代码编程、通用概念与原理、生活常识、历史地理等，"
                "且【不得】涉及任何法律、法规、法条、诉讼、合同、维权等法律话题。")
LAW_DESC = ("内容必须与法律相关：可以是查询某部法律或某条文（例如《民法典》第X条）、"
            "解释法律概念术语（如正当防卫、善意取得、表见代理）、具体纠纷如何维权、"
            "诉讼与执行程序，或合同、婚姻家庭、劳动争议、消费维权、公司、交通、房屋租赁等场景的法律问题。")

GENERAL_TOPICS = ["数学运算", "Python 编程", "Java 编程", "数据结构与算法", "网络与 HTTP",
                  "数据库与 SQL", "操作系统", "人工智能/机器学习概念", "前端/后端开发",
                  "日常健康", "饮食营养", "天气与地理", "历史常识", "物理化学常识",
                  "生活小技巧", "工具软件使用", "英语翻译", "办公软件", "旅行出行",
                  "汽车常识", "电子产品", "游戏娱乐", "体育常识", "理财常识（非法律）"]
LAW_TOPICS = ["民法典合同编", "婚姻家庭与继承", "劳动争议与工伤", "消费者权益", "交通事故与赔偿",
              "房屋租赁与买卖", "公司法与股东", "借贷与担保", "侵权与人身损害", "刑事责任与量刑",
              "行政处罚与复议", "民事诉讼程序", "商标/专利/著作权", "个人信息与数据合规",
              "劳动争议仲裁", "征地拆迁", "医疗纠纷", "食品安全", "环境保护与排污",
              "电信网络诈骗", "法律援助", "治安管理处罚", "仲裁与调解", "诉讼时效"]

SYSTEM_PROMPT = "你是严格按要求输出 JSONL 的中文数据生成器，只输出 JSON 行，绝不输出任何多余文字。"


def log(msg):
    print(msg, flush=True)


def norm_key(q):
    return re.sub(r"\s+", "", q or "")


def load_llm():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, "config.ini"), encoding="utf-8")
    model = cfg.get("llm", "model", fallback="qwen3.8-27b")
    key = cfg.get("llm", "dashscope_api_key", fallback="")
    base = cfg.get("llm", "dashscope_base_url", fallback="")
    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        for line in open(env_file, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "DASHSCOPE_API_KEY":
                key = v
            elif k == "DASHSCOPE_BASE_URL":
                base = v
    return model, key, base


def user_prompt(label, n, topics):
    desc = GENERAL_DESC if label == "通用知识" else LAW_DESC
    topic_hint = "、".join(topics)
    return (
        f"请生成 {n} 条「{label}」类别的中文用户查询样本，每条一行 JSON，"
        f"格式为 {{\"query\": \"……\", \"label\": \"{label}\"}}。\n"
        "要求：\n"
        f"1. label 字段必须严格等于「{label}」，不要出现其他标签。\n"
        f"2. {desc}\n"
        "3. 查询口语化、简短（约 4~30 个字），覆盖不同主题、不同问法与不同长度，尽量避免重复。\n"
        "4. 只输出 JSON 行，不要编号、不要解释、不要代码块、不要空行。\n"
        f"5. 尽量围绕这些主题方向发散：{topic_hint}。\n"
        f"现在开始输出 {n} 行："
    )


def parse_jsonl(text, want_label):
    out = []
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip().strip(",").strip()
        if not line or line.startswith("```"):
            continue
        obj = None
        try:
            obj = json.loads(line)
        except Exception:
            m = re.search(r"\{.*\}", line)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    obj = None
        if isinstance(obj, dict) and obj.get("label") == want_label:
            q = str(obj.get("query", "")).strip()
            if q:
                out.append(q)
    return out


def gen_batch(client, model, label, n, rng):
    topics = rng.sample(GENERAL_TOPICS if label == "通用知识" else LAW_TOPICS, k=6)
    prompt = user_prompt(label, n, topics)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=4000,
            )
            return parse_jsonl(resp.choices[0].message.content or "", label)
        except Exception as e:
            if attempt == 2:
                log(f"    [warn] {label} 批次失败: {str(e)[:120]}")
                return []
    return []


def load_existing():
    data = []
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("label") in LABELS and o.get("query"):
                    data.append({"query": str(o["query"]).strip(), "label": o["label"]})
    return data


def fill_label(client, model, label, target, batch, executor, seen, results, lock):
    rng = random.Random(abs(hash(label)) & 0xFFFF)
    rounds = 0
    while len(results[label]) < target and rounds < 60:
        rounds += 1
        need = target - len(results[label])
        nbatch = min(12, max(1, math.ceil(need / batch * 1.6)))
        futures = [executor.submit(gen_batch, client, model, label, batch, rng)
                   for _ in range(nbatch)]
        got = 0
        for fut in as_completed(futures):
            for q in fut.result():
                k = norm_key(q)
                with lock:
                    if k in seen or len(results[label]) >= target:
                        continue
                    seen.add(k)
                    results[label].append(q)
                    got += 1
        log(f"    {label}: +{got} -> {len(results[label])}/{target}")
        save(results)  # 每轮落盘，断点续跑最细到一轮


def save(results):
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for lab in LABELS:
            for q in results[lab]:
                fo.write(json.dumps({"query": q, "label": lab}, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT_FILE)


def main():
    ap = argparse.ArgumentParser(description="法律领域二分类训练数据生成")
    ap.add_argument("--per-class", type=int, default=2500, help="每个类别生成条数")
    ap.add_argument("--batch", type=int, default=60, help="单次请求生成条数")
    ap.add_argument("--workers", type=int, default=4, help="并发请求数")
    ap.add_argument("--rebuild", action="store_true", help="忽略已有产物，重新生成")
    args = ap.parse_args()

    os.makedirs(CLASSIFY_DIR, exist_ok=True)
    if args.rebuild and os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    existing = load_existing()
    seen = {norm_key(x["query"]) for x in existing}
    results = {lab: [x["query"] for x in existing if x["label"] == lab] for lab in LABELS}
    log(f"目标: 每类 {args.per_class} 条；已有 通用知识 {len(results['通用知识'])}、法律咨询 {len(results['法律咨询'])}")

    if all(len(results[lab]) >= args.per_class for lab in LABELS):
        log("已达目标，无需生成。用 --rebuild 可重来。")
        save(results)
        return

    model, key, base = load_llm()
    if not key or "your-key" in key:
        log("未找到可用的 LLM API Key（config.ini / .env），退出。")
        sys.exit(1)
    log(f"LLM: model={model} base_url={base}")
    client = OpenAI(api_key=key, base_url=base)
    lock = threading.Lock()

    # 注意: fill_label 会向 executor 提交任务并在主线程等待结果,
    # 因此两个类别必须顺序补齐(不能把 fill_label 本身也丢进同一个池, 否则会死锁)。
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for lab in LABELS:
            fill_label(client, model, lab, args.per_class, args.batch,
                       executor, seen, results, lock)
            save(results)  # 边生成边落盘，避免中途失败白跑

    save(results)
    total = sum(len(results[lab]) for lab in LABELS)
    lengths = [len(q) for lab in LABELS for q in results[lab]]
    log("\n[汇总]")
    for lab in LABELS:
        log(f"    {lab}: {len(results[lab])} 条")
    log(f"    合计: {total} 条")
    if lengths:
        log(f"    query 长度: 最短 {min(lengths)} / 最长 {max(lengths)} / 平均 {sum(lengths)/len(lengths):.1f} 字")
    log(f"    产物: {OUT_FILE}")


if __name__ == "__main__":
    main()
