# -*- coding: utf-8 -*-
"""
法律 FAQ 抽样（P0）
==================
把 Brench 法律问答（合成风格，答案自带《X法》第Y条 引用）按「领域均衡 + 法条可核对」
抽成 law_faq 可直接入库的 CSV，替换原来那批「为了迁就 97 部语料筛出来」的问答。
筛选与配额规则:
  1) 答案里引用的法律必须都在当前语料（law_data）里；
  2) 引用的条文号必须真实存在于该法律的条号集合里（按 law_data 实际内容核对）；
  3) 问题去重（忽略空白后比对）；
  4) 按领域配额抽样，配额不足的领域由剩余样本补齐（总量保持 n 条）。
用法:
  python tools/sample_law_faq.py --stats              # 只统计，不写文件
  python tools/sample_law_faq.py                      # 抽 20000 条 -> mysql_qa/data/law_qa.csv
  python tools/sample_law_faq.py --n 20000 --seed 0
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.law_parser import norm_law_name, cn2int  # noqa: E402

SRC_DEFAULT = os.path.join(ROOT, "rag_qa", "data", "law_eval", "qa", "laws_data_50k_test.jsonl")
LAW_DATA_DIR = os.path.join(ROOT, "rag_qa", "data", "law_data")
OUT_DEFAULT = os.path.join(ROOT, "mysql_qa", "data", "law_qa.csv")

BOOK_RE = re.compile(r"《([^》]{2,30})》")
CITE_RE = re.compile(r"《([^》]{2,30})》第([一二三四五六七八九十百千万零〇两0-9]{1,8})条")
ART_RE = re.compile(r"^第([一二三四五六七八九十百千万零〇两0-9]{1,8})条")

# 八大领域（对齐 1.20 模板的领域划分）＋兜底
DOMAIN_RULES = [
    ("劳动法务", ["劳动法", "劳动合同法", "社会保险法", "劳动争议调解仲裁法", "就业促进法", "工会法"]),
    ("公司法务", ["公司法", "合伙企业法", "个人独资企业法", "企业破产法", "证券法", "保险法",
                  "票据法", "商业银行法", "外商投资法", "注册会计师法", "期货和衍生品法"]),
    ("知识产权", ["专利法", "商标法", "著作权法", "反不正当竞争法"]),
    ("刑事法律", ["刑法", "刑事诉讼法", "反有组织犯罪法", "治安管理处罚法", "反电信网络诈骗法",
                  "反洗钱法", "禁毒法"]),
    ("行政法务", ["行政处罚法", "行政复议法", "行政诉讼法", "行政许可法", "行政强制法",
                  "国家赔偿法", "公务员法", "监察法"]),
    ("婚姻家事", ["反家庭暴力法", "妇女权益保障法", "未成年人保护法", "家庭教育促进法",
                  "母婴保健法", "老年人权益保障法", "残疾人保障法"]),
    ("合同与商事", ["招标投标法", "拍卖法", "电子签名法", "海商法", "海事诉讼特别程序法",
                    "民用航空法", "港口法"]),
    ("侵权与消费", ["消费者权益保护法", "产品质量法", "道路交通安全法",
                    "基本医疗卫生与健康促进法"]),
]
FALLBACK_DOMAIN = "其他法律法规"
CIVIL_CODE_DOMAIN = "民事综合"

# 民法典按问题关键词细分领域
CIVIL_CODE_RULES = [
    ("婚姻家事", ["离婚", "结婚", "婚姻", "抚养", "赡养", "继承", "遗嘱", "遗产", "夫妻", "彩礼", "收养"]),
    ("合同与商事", ["合同", "违约", "买卖", "借款", "租赁", "定金", "承揽", "保证", "欠款", "订购", "供货"]),
    ("侵权与消费", ["侵权", "损害", "赔偿", "过错", "名誉", "隐私", "肖像"]),
]

DOMAIN_ORDER = [d for d, _ in DOMAIN_RULES] + [CIVIL_CODE_DOMAIN, FALLBACK_DOMAIN]

# 源数据的引用标签是「类别-法律名」或「民法典-XX编」，这里还原成语料里的法律名，
# 顺便把民法典的编直接当领域提示用。
CIVIL_CODE_SUB = {
    "合同编": "合同与商事",
    "物权编": "物权与担保",
    "侵权责任编": "侵权与消费",
    "婚姻家庭编": "婚姻家事",
    "继承编": "婚姻家事",
    "人格权编": "人格权与个人信息",
    "总则": CIVIL_CODE_DOMAIN,
}


def parse_cited(label, corpus):
    """把引用标签还原成 (语料中的法律名 | None, 领域提示 | None)。"""
    s = norm_law_name(label)
    if "民法典" in s:
        for sub, dom in CIVIL_CODE_SUB.items():
            if sub in s:
                return "民法典", dom
        return "民法典", None
    if "-" in s:
        parts = [p.strip() for p in s.split("-") if p.strip()]
        for p in parts:
            if p in corpus:
                return p, None
        for p in parts:
            for k in corpus:
                if len(k) >= 4 and p.endswith(k):
                    return k, None
        return None, None
    return (s, None) if s in corpus else (None, None)


def log(msg):
    print(msg, flush=True)


def load_corpus():
    """返回 (语料法律名集合, {法律名: 条文号集合}, 总条数)。"""
    names, numbers, total = set(), {}, 0
    for fn in os.listdir(LAW_DATA_DIR):
        if not fn.endswith(".txt"):
            continue
        law = norm_law_name(os.path.splitext(fn)[0])
        nums = set()
        with open(os.path.join(LAW_DATA_DIR, fn), encoding="utf-8") as f:
            for line in f:
                m = ART_RE.match(line.strip())
                if m:
                    v = cn2int(m.group(1))
                    if v is not None:
                        nums.add(v)
        names.add(law)
        numbers[law] = nums
        total += len(nums)
    return names, numbers, total


def domain_of(cited, hints, text):
    """判定领域：先看引用自带的编提示，再按法律名匹配，最后看民法典问题的关键词。"""
    if hints:
        return sorted(hints)[0]
    for law in cited:
        for domain, keys in DOMAIN_RULES:
            if any(k in law for k in keys):
                return domain
    if any("民法典" in law for law in cited):
        for domain, keys in CIVIL_CODE_RULES:
            if any(k in text for k in keys):
                return domain
        return CIVIL_CODE_DOMAIN
    return FALLBACK_DOMAIN


def clean_labels(text, corpus):
    """把源数据的《类别-法律名》《民法典-XX编》统一成知识库里的《法律名》，正文更干净。"""
    def repl(m):
        law, _h = parse_cited(m.group(1), corpus)
        return f"《{law}》" if law else m.group(0)
    return BOOK_RE.sub(repl, text)


def main():
    ap = argparse.ArgumentParser(description="法律 FAQ 抽样 -> law_qa.csv")
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    corpus, numbers, total_articles = load_corpus()
    log(f"语料：{len(corpus)} 部法律 / {total_articles} 条法条")
    stat = Counter()
    cited_counter, domain_pool = Counter(), {}
    seen = set()
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stat["源总行数"] += 1
            o = json.loads(line)
            q = clean_labels(o.get("question", ""), corpus)
            a = clean_labels(o.get("answer", ""), corpus)
            labels = BOOK_RE.findall(a)
            if not labels:
                stat["无法律引用"] += 1
                continue
            parsed = [parse_cited(x, corpus) for x in labels]
            if any(law is None for law, _ in parsed):
                stat["引用了语料外的法律"] += 1
                continue
            cited = {law for law, _ in parsed}
            hints = {h for _, h in parsed if h}
            bad = False
            for law, num in CITE_RE.findall(a):
                law_n, _h = parse_cited(law, corpus)
                v = cn2int(num)
                if law_n in numbers and (v is None or v not in numbers[law_n]):
                    bad = True
                    break
            if bad:
                stat["条号不存在于语料"] += 1
                continue
            key = re.sub(r"\s+", "", q)
            if key in seen:
                stat["重复问题"] += 1
                continue
            seen.add(key)
            for law in cited:
                cited_counter[law] += 1
            d = domain_of(cited, hints, q + a)
            domain_pool.setdefault(d, []).append({"法律领域": d, "问题": q, "答案": a})

    log(f"\n== 筛选统计（源：{os.path.basename(args.src)}）==")
    for k in ["源总行数", "无法律引用", "引用了语料外的法律", "条号不存在于语料", "重复问题"]:
        log(f"  {k}: {stat[k]}")
    usable = sum(len(v) for v in domain_pool.values())
    log(f"  可用于抽样: {usable}")
    log("\n== 各领域可用量 ==")
    for d in DOMAIN_ORDER:
        if domain_pool.get(d):
            log(f"  {d}: {len(domain_pool[d])}")
    log("\n== 被引用最多的法律 Top 15 ==")
    for law, c in cited_counter.most_common(15):
        log(f"  {c:6d}  {law}")
    log(f"  覆盖法律部数: {len(cited_counter)}")
    if args.stats:
        log("\n[--stats] 不写文件，退出")
        return

    rnd = random.Random(args.seed)
    quota = max(1, args.n // len(DOMAIN_ORDER))
    chosen, pool = [], {}
    for d in DOMAIN_ORDER:
        items = list(domain_pool.get(d, []))
        rnd.shuffle(items)
        chosen.extend(items[:quota])
        pool[d] = items[quota:]
    log(f"\n首轮 {len(DOMAIN_ORDER)} 领域 × {quota} 条 = {len(chosen)} 条")
    if len(chosen) < args.n:
        rest = []
        for d in DOMAIN_ORDER:
            rnd.shuffle(pool[d])
            rest.extend(pool[d])
        rnd.shuffle(rest)
        add = min(len(rest), args.n - len(chosen))
        chosen.extend(rest[:add])
        log(f"  剩余样本补齐 {add} 条")
    chosen = chosen[: args.n]
    rnd.shuffle(chosen)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["法律领域", "问题", "答案"])
        w.writeheader()
        w.writerows(chosen)
    c = Counter(x["法律领域"] for x in chosen)
    log(f"\n已写入 {len(chosen)} 条 -> {os.path.relpath(args.out, ROOT)}")
    for d in DOMAIN_ORDER:
        if c.get(d):
            log(f"  {d}: {c[d]}")
    log("完成")


if __name__ == "__main__":
    main()
