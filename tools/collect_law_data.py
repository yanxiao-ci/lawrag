#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
法律方向数据采集脚本
====================
对应《法律方向改造方案书》§3.1「最小可用集」：把项目从 IT 培训场景改造为
法律法规智能问答助手所需的外部数据一次性拉齐并落位。

采集清单:
  [1] 主语料     KuugoRen/Chinese_Law                       (ModelScope, 97 部法律 .txt)
                 -> rag_qa/data/law_data/                  (Milvus 知识库 + MySQL 条文表共用)
  [2] 检索评测   ByronLeeee/CN-Law-Query-Retrieval-Dataset (ModelScope, {anchor,positive,negative})
                 -> rag_qa/data/law_eval/retrieval/finetune_dataset.jsonl
  [3] 问答评测   Brench/chinese_law_data_rag_ft             (HuggingFace 镜像, 测试集 jsonl)
                 -> rag_qa/data/law_eval/qa/laws_data_50k_test.jsonl
  [4] 编/章层级  Dusker/chinese-laws-pretrain               (HuggingFace 镜像, 按编分文件)  [可选 --with-chapter]
                 -> rag_qa/data/law_chapter/

用法:
  python tools/collect_law_data.py                  # 采集 [1]~[3]（默认）
  python tools/collect_law_data.py --with-chapter   # 额外采集 [4]
  python tools/collect_law_data.py --verify-only    # 只跑主语料的解析校验
  python tools/collect_law_data.py --force          # 忽略已存在文件，重新下载
  python tools/collect_law_data.py --rebuild-subsets  # 只按当前语料重跑评测集裁剪（语料变更后用）

备注:
  - 本机直连 huggingface.co 不通，默认走 hf-mirror.com；可用环境变量 HF_ENDPOINT 覆盖。
  - 大体积评测集/层级语料已写入 .gitignore，不进 git。
"""

import argparse
import json
import os
import re
import sys
from urllib.parse import quote

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "rag_qa", "data")
LAW_DATA_DIR = os.path.join(DATA_DIR, "law_data")
EVAL_RETRIEVAL_DIR = os.path.join(DATA_DIR, "law_eval", "retrieval")
EVAL_QA_DIR = os.path.join(DATA_DIR, "law_eval", "qa")
CHAPTER_DIR = os.path.join(DATA_DIR, "law_chapter")

MS_API = "https://modelscope.cn/api/v1/datasets"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

DS_LAW = "KuugoRen/Chinese_Law"
DS_RETRIEVAL = "ByronLeeee/CN-Law-Query-Retrieval-Dataset"
DS_QA = "Brench/chinese_law_data_rag_ft"
DS_CHAPTER = "Dusker/chinese-laws-pretrain"

RETRIEVAL_FILE = "finetune_dataset.jsonl"
QA_TEST_FILE = "laws_data_50k_test.jsonl"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (lawrag law-data collector)"})


def log(msg):
    print(msg, flush=True)


def human(nbytes):
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024


def download(url, dst, force=False):
    """流式下载到 dst，返回 (path, bytes_written)。已存在则跳过。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and not force:
        size = os.path.getsize(dst)
        log(f"    [skip] 已存在 {os.path.basename(dst)} ({human(size)})")
        return dst, 0
    tmp = dst + ".part"
    with SESSION.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
        os.replace(tmp, dst)
    tail = f" / {human(total)}" if total else ""
    log(f"    [ok]   {os.path.basename(dst)} ({human(got)}{tail})")
    return dst, got


def ms_list_files(dataset):
    url = f"{MS_API}/{dataset}/repo/tree?Revision=master&Recursive=1"
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    files = r.json()["Data"]["Files"]
    return [f for f in files if f.get("Type") == "blob"]


def ms_download(dataset, path, dst, force=False):
    url = f"{MS_API}/{dataset}/repo?Revision=master&FilePath={quote(path)}"
    return download(url, dst, force=force)


def hf_list_files(repo):
    url = f"{HF_ENDPOINT}/api/datasets/{repo}"
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    return [s["rfilename"] for s in r.json().get("siblings", [])]


def hf_download(repo, path, dst, force=False):
    url = f"{HF_ENDPOINT}/datasets/{repo}/resolve/main/{quote(path)}"
    return download(url, dst, force=force)


# ----------------------------------------------------------------------------
# [1] 主语料
# ----------------------------------------------------------------------------
def collect_corpus(force=False):
    log(f"\n[1] 主语料 {DS_LAW} (ModelScope) -> {LAW_DATA_DIR}")
    blobs = ms_list_files(DS_LAW)
    txts = [b for b in blobs if b["Path"].lower().endswith(".txt")]
    log(f"    远端 .txt 文件数: {len(txts)}")
    for b in txts:
        name = b["Path"].split("/")[-1]
        ms_download(DS_LAW, b["Path"], os.path.join(LAW_DATA_DIR, name), force=force)
    return len(txts)


# ----------------------------------------------------------------------------
# [2] 检索评测
# ----------------------------------------------------------------------------
def collect_retrieval(force=False):
    log(f"\n[2] 检索评测 {DS_RETRIEVAL} (ModelScope)")
    blobs = ms_list_files(DS_RETRIEVAL)
    target = [b for b in blobs if b["Path"].endswith(RETRIEVAL_FILE)]
    if not target:
        raise RuntimeError(f"远端未找到 {RETRIEVAL_FILE}: {[b['Path'] for b in blobs]}")
    dst = os.path.join(EVAL_RETRIEVAL_DIR, RETRIEVAL_FILE)
    ms_download(DS_RETRIEVAL, target[0]["Path"], dst, force=force)
    return dst


# ----------------------------------------------------------------------------
# [3] 问答评测
# ----------------------------------------------------------------------------
def collect_qa(force=False):
    log(f"\n[3] 问答评测 {DS_QA} (HuggingFace 镜像)")
    dst = os.path.join(EVAL_QA_DIR, QA_TEST_FILE)
    hf_download(DS_QA, QA_TEST_FILE, dst, force=force)
    return dst


# ----------------------------------------------------------------------------
# [4] 编/章层级（可选）
# ----------------------------------------------------------------------------
def collect_chapter(force=False):
    log(f"\n[4] 编/章层级 {DS_CHAPTER} (HuggingFace 镜像) -> {CHAPTER_DIR}")
    files = [f for f in hf_list_files(DS_CHAPTER) if f.lower().endswith(".json")]
    log(f"    远端 .json 文件数: {len(files)}")
    for f in files:
        name = f.split("/")[-1]
        hf_download(DS_CHAPTER, f, os.path.join(CHAPTER_DIR, name), force=force)
    return len(files)


# ----------------------------------------------------------------------------
# 解析校验（对齐方案书 §1.1）
# ----------------------------------------------------------------------------
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
MAIN_RE = re.compile(
    r"^《(?P<law>[^》]+)》第(?P<num>[一二三四五六七八九十百千万零〇两0-9]+)条规定，(?P<text>.*)$")
FALLBACK_RE = re.compile(
    r"^第(?P<num>[一二三四五六七八九十百千万零〇两0-9]+)条[\s　]*(?P<text>.*)$")

# 评测集里法律名有全称/简称两种写法，统一去掉"中华人民共和国"前缀再比对
NUM_RE = r"[一二三四五六七八九十百千万零〇两0-9]+"
TITLE_RE = re.compile(r"title:\s*(.+?)\s*第" + NUM_RE + r"条\s*\|")
BOOK_RE = re.compile(r"《([^》]+)》")


def norm_law(name):
    return name.replace("中华人民共和国", "").strip()


def cn2int(s):
    if s.isdigit():
        return int(s)
    total = section = number = 0
    for ch in s:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            if unit == 10000:
                total += (section + number) * unit
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number


def verify_corpus():
    log("\n[校验] 解析 rag_qa/data/law_data (对齐方案书 §1.1)")
    if not os.path.isdir(LAW_DATA_DIR):
        log("    law_data 目录不存在，跳过")
        return None
    txt_files = sorted(f for f in os.listdir(LAW_DATA_DIR) if f.lower().endswith(".txt"))
    total_lines = parsed = num_ok = 0
    fallback_laws = {}
    for fn in txt_files:
        law_name = os.path.splitext(fn)[0]
        with open(os.path.join(LAW_DATA_DIR, fn), encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                total_lines += 1
                m = MAIN_RE.match(line)
                if m is None:
                    m = FALLBACK_RE.match(line)
                    if m is not None:
                        fallback_laws[law_name] = fallback_laws.get(law_name, 0) + 1
                if m is None:
                    continue
                parsed += 1
                if cn2int(m.group("num")) is not None:
                    num_ok += 1
    log(f"    法律文件数        : {len(txt_files)}")
    log(f"    非空总行数        : {total_lines}")
    rate = (parsed / total_lines * 100) if total_lines else 0
    log(f"    成功解析为法条    : {parsed} ({rate:.2f}%)")
    ok_rate = (num_ok / parsed * 100) if parsed else 0
    log(f"    条文号可转数值    : {num_ok} / {parsed} ({ok_rate:.1f}%)")
    if fallback_laws:
        log("    兜底规则命中(按法律):")
        for k, v in sorted(fallback_laws.items(), key=lambda x: -x[1]):
            log(f"        {k}: {v}")
    return {"files": len(txt_files), "lines": total_lines, "parsed": parsed,
            "num_ok": num_ok, "fallback": fallback_laws}


def build_incorpus_subsets():
    """评测集按当前主语料（rag_qa/data/law_data）裁剪，生成「语料内」子集。

    ByronLeeee 覆盖 309 部法律、Brench 覆盖 753 部，都远超本项目语料。
    若直接全量评测，大量样本的正确答案根本不在知识库里，Recall@k 恒为 0，
    指标会失真。这里只保留「正确答案落在当前语料之内」的样本。
    语料扩容后需重跑本函数（本脚本 main 末尾会自动调用）。
    """
    if not os.path.isdir(LAW_DATA_DIR):
        return
    laws = {norm_law(os.path.splitext(f)[0])
            for f in os.listdir(LAW_DATA_DIR) if f.lower().endswith(".txt")}
    log(f"\n[裁剪] 生成「语料内」评测子集（当前语料 {len(laws)} 部法律）")

    src = os.path.join(EVAL_RETRIEVAL_DIR, RETRIEVAL_FILE)
    if os.path.exists(src):
        dst = src.replace(".jsonl", "_incorpus.jsonl")
        kept = total = 0
        with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                line = line.strip()
                if not line:
                    continue
                total += 1
                o = json.loads(line)
                m = TITLE_RE.search(o.get("positive", ""))
                if m and norm_law(m.group(1)) in laws:
                    fo.write(json.dumps(o, ensure_ascii=False) + "\n")
                    kept += 1
        rate = kept / total * 100 if total else 0
        log(f"    {os.path.basename(dst)}: {kept}/{total} ({rate:.1f}%)")

    src = os.path.join(EVAL_QA_DIR, QA_TEST_FILE)
    if os.path.exists(src):
        dst = src.replace(".jsonl", "_incorpus.jsonl")
        kept = total = 0
        with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                line = line.strip()
                if not line:
                    continue
                total += 1
                o = json.loads(line)
                refs = {norm_law(r) for r in BOOK_RE.findall(o.get("answer", ""))}
                if refs and refs <= laws:
                    fo.write(json.dumps(o, ensure_ascii=False) + "\n")
                    kept += 1
        rate = kept / total * 100 if total else 0
        log(f"    {os.path.basename(dst)}: {kept}/{total} ({rate:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="法律方向数据采集（方案书 §3.1）")
    ap.add_argument("--with-chapter", action="store_true", help="额外采集编/章层级语料")
    ap.add_argument("--verify-only", action="store_true", help="只跑解析校验")
    ap.add_argument("--force", action="store_true", help="忽略已存在文件，重新下载")
    ap.add_argument("--rebuild-subsets", action="store_true",
                    help="只按当前 law_data 重跑评测集裁剪（语料扩容后要用这个，避免重新下载旧语料）")
    args = ap.parse_args()

    if args.rebuild_subsets:
        # 语料已经由 tools/expand_law_corpus.py 生成，这里只按新语料重跑裁剪与校验，
        # 不去碰 law_data（否则会把旧的 97 部 KuugoRen 语料再拉回来混进语料目录）。
        verify_corpus()
        build_incorpus_subsets()
        log("\n完成（只重建了评测子集）。")
        return

    if not args.verify_only:
        collect_corpus(force=args.force)
        collect_retrieval(force=args.force)
        collect_qa(force=args.force)
        if args.with_chapter:
            collect_chapter(force=args.force)

    verify_corpus()
    build_incorpus_subsets()
    log("\n完成。下一步见《法律方向改造方案书》§7 动手顺序。")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        log(f"网络请求失败: {e}")
        sys.exit(1)
