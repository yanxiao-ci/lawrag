# -*- coding: utf-8 -*-
"""
法条语料扩容（P0 + P1 数据准备）
================================
把「全量法规库」（twang2218/chinese-law-and-regulations = 国家法律法规数据库口径）
导出成项目现有 pipeline 直接可用的 rag_qa/data/law_data/*.txt。

选取口径：
  1) type = 法律 且 status = 有效        -> 现行有效法律全量（P0）
  2) type = 宪法 中的「2018 年修正文本」 -> 现行宪法
  3) --with-history 指定的法律，额外导出其被修订/废止的上一版（P1 时效演示用）

同时生成 rag_qa/data/law_meta.json：
  { 法律名: {type, status, publish_date, effective_date, office, n_articles, chapter_ranges} }
  chapter_ranges = [{"path": "第一编 总则 第一章 基本规定", "from": 1, "to": 7}, ...]
  供 MySQL 法条表填「编/章/节」列、检索时做时效过滤与层级展示。

用法:
  python tools/expand_law_corpus.py                     # 从 _law_src/data 读，导出 law_data
  python tools/expand_law_corpus.py --keep-old          # 不备份旧语料（重新生成时用）
  python tools/expand_law_corpus.py --with-history 行政处罚法,商标法
  python tools/expand_law_corpus.py --no-history        # 不带历史版本
  python tools/expand_law_corpus.py --dry-run           # 只统计不写文件
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base.law_parser import cn2int, parse_law_dir, parse_law_file  # noqa: E402

DEFAULT_SRC_DIR = os.path.join(ROOT, "_law_src", "data")
LAW_DATA_DIR = os.path.join(ROOT, "rag_qa", "data", "law_data")
BACKUP_ROOT = os.path.join(ROOT, "rag_qa", "data")
META_OUT = os.path.join(ROOT, "rag_qa", "data", "law_meta.json")

ARTICLE_LINE = re.compile(
    r"^第(?P<num>[一二三四五六七八九十百千万零〇两0-9]{1,8})条"
    r"(?:之(?P<suf>[一二三四五六七八九十]{1,3}))?[\s\u3000]*(?P<body>.*)$")
OUTLINE_LINE = re.compile(
    r"^(第[一二三四五六七八九十百千万零〇两0-9]{1,6}(?:分)?[编章节])[\s\u3000]*(.*)$")

CONSTITUTION_TITLE = "中华人民共和国宪法（2018年修正文本）"
HISTORY_LAWS_DEFAULT = ["中华人民共和国行政处罚法", "中华人民共和国商标法",
                        "中华人民共和国反间谍法", "中华人民共和国专利法"]
HISTORY_STATUS = {"已修改", "已废止"}
# 「本法自 2026年1月1日 起施行」——用于给补充语料推断施行日期
EFFECTIVE_RE = re.compile(r"本法自\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?起施行")


def adopt_unmanaged(law_dir, generated, meta):
    """把 law_dir 里不是本次生成的 .txt（如 KuugoRen 补充的法律）也纳入元数据。
    否则元数据与语料会脱节：这些文件会在入库时被当成「有效」但没有任何时效/层级信息。
    """
    adopted = []
    for fn in sorted(os.listdir(law_dir)) if os.path.isdir(law_dir) else []:
        if not fn.endswith(".txt"):
            continue
        stem = os.path.splitext(fn)[0]
        if stem in generated:
            continue
        path = os.path.join(law_dir, fn)
        try:
            rows = parse_law_file(path)
        except Exception:
            rows = []
        if not rows:
            continue
        eff = ""
        for _law, _no, _num, text in rows:
            m = EFFECTIVE_RE.search(text or "")
            if m:
                eff = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        meta[stem] = {
            "type": "法律",
            "status": "有效",
            "is_history": False,
            "source": "kuugoren",
            "publish_date": "",
            "effective_date": eff,
            "office": "",
            "n_articles": len(rows),
            "chapter_ranges": [],
        }
        adopted.append((stem, len(rows), eff))
    return adopted


def log(msg):
    print(msg, flush=True)


def find_src(explicit):
    if explicit:
        return explicit
    if not os.path.isdir(DEFAULT_SRC_DIR):
        raise SystemExit(f"找不到默认数据源目录 {DEFAULT_SRC_DIR}，请用 --src 指定 parquet")
    cands = []
    for fn in os.listdir(DEFAULT_SRC_DIR):
        p = os.path.join(DEFAULT_SRC_DIR, fn)
        if fn.endswith(".parquet") and os.path.getsize(p) > 1024:
            cands.append(p)
    if not cands:
        raise SystemExit(f"{DEFAULT_SRC_DIR} 下没有可用的 parquet，请用 --src 指定")
    return max(cands, key=os.path.getsize)


def split_articles(content):
    """切出 [法条行...]，并按 编/章/节 标题统计区间 -> (articles, chapter_ranges)。"""
    articles, segs, stack, cur = [], [], {}, None
    for raw in (content or "").splitlines():
        line = raw.strip().strip("\u3000").strip()
        if not line or line.startswith(">"):
            continue
        line = line.replace("\u3000", " ").strip()
        m = ARTICLE_LINE.match(line)
        if m:
            body = m.group("body").strip()
            no = f"第{m.group('num')}条" + (f"之{m.group('suf')}" if m.group("suf") else "")
            articles.append(f"{no}\u3000{body}" if body else no)
            num = cn2int(m.group("num"))
            path = " ".join(stack[k] for k in "编章节" if k in stack)
            if cur is None or cur["path"] != path:
                cur = {"path": path, "from": num, "to": num}
                segs.append(cur)
            else:
                cur["to"] = num
            continue
        o = OUTLINE_LINE.match(line)
        if o:
            unit = "节" if o.group(1).endswith("节") else ("编" if o.group(1).endswith("编") else "章")
            head = f"{o.group(1)} {o.group(2)}".strip()
            if unit == "编":
                stack.pop("章", None)
                stack.pop("节", None)
            elif unit == "章":
                stack.pop("节", None)
            stack[unit] = head
    return articles, segs


def safe_name(title):
    return re.sub(r'[\\/:*?"<>|]', "／", title).strip()


def select_rows(df, history_laws):
    """选现行语料 + 指定法律的历史版本。"""
    laws = df[(df["type"] == "法律") & (df["status"] == "有效")].copy()
    const = df[(df["type"] == "宪法") & (df["title"] == CONSTITUTION_TITLE)].copy()
    if const.empty:
        const = df[(df["type"] == "宪法") & (df["title"].str.contains("修正文本", na=False))].copy()
    rows = pd.concat([const, laws], ignore_index=True)

    history = []
    for title in history_laws or []:
        sub = df[(df["title"] == title) & (df["status"].isin(HISTORY_STATUS))].copy()
        if sub.empty:
            log(f"      [历史] {title}: 库里没有已修改/已废止版本")
            continue
        sub["_d"] = pd.to_datetime(sub["publish_date"], errors="coerce")
        r = sub.sort_values("_d").iloc[-1]
        history.append(r)
    if history:
        rows = pd.concat([rows, pd.DataFrame(history)], ignore_index=True)
    return rows, len(history)


def main():
    ap = argparse.ArgumentParser(description="法条语料扩容：全量法规库 -> law_data")
    ap.add_argument("--src", default=None, help="全量法规 parquet 路径")
    ap.add_argument("--keep-old", action="store_true", help="不备份旧语料")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    ap.add_argument("--with-history", default=",".join(HISTORY_LAWS_DEFAULT),
                    help="额外导出这些法律的历史版本（逗号分隔法律全名）")
    ap.add_argument("--no-history", action="store_true", help="不导出历史版本")
    args = ap.parse_args()

    history_laws = [] if args.no_history else [x.strip() for x in args.with_history.split(",") if x.strip()]

    src = find_src(args.src)
    log(f"[1/5] 数据源: {src} ({os.path.getsize(src) / 1024 / 1024:.1f} MB)")
    df = pd.read_parquet(src)
    log(f"      全量 {len(df)} 篇")

    rows, n_history = select_rows(df, history_laws)
    log(f"[2/5] 选中 {len(rows)} 篇（宪法 1 + 现行有效法律 {len(rows) - 1 - n_history} + 历史版本 {n_history}）")

    files, total_articles, empty, meta = {}, 0, [], {}
    skipped_nested = []
    for _, r in rows.iterrows():
        articles, segs = split_articles(r["content"])
        title = safe_name(r["title"])
        is_history = r["status"] in HISTORY_STATUS
        if r["type"] == "宪法":
            title = "中华人民共和国宪法"
        if is_history:
            year = str(r["publish_date"])[:4]
            if not year.isdigit():
                year = str(r["effective_date"])[:4]
            title = f"{title}（{year}年版）"
        # 跳过「批准类决议」：标题里嵌了《》的文件，引用格式会出现嵌套书名号，而且它们不是独立法律
        if "《" in title or "》" in title:
            skipped_nested.append(title)
            continue
        if not articles:
            empty.append(title)
            continue
        files[title] = articles
        total_articles += len(articles)
        meta[title] = {
            "type": r["type"],
            "status": r["status"],
            "is_history": bool(is_history),
            "publish_date": str(r["publish_date"])[:10] if pd.notna(r["publish_date"]) else "",
            "effective_date": str(r["effective_date"])[:10] if pd.notna(r["effective_date"]) else "",
            "office": r["office"] if pd.notna(r["office"]) else "",
            "n_articles": len(articles),
            "chapter_ranges": segs,
        }
    log(f"      法条合计 {total_articles} 条；无法条被跳过 {len(empty)} 篇")
    if skipped_nested:
        log(f"      标题含书名号被跳过 {len(skipped_nested)} 篇（批准类决议，非独立法律）: "
            f"{[t[:24] + '…' for t in skipped_nested]}")
    with_chap = sum(1 for v in meta.values() if any(s["path"] for s in v["chapter_ranges"]))
    log(f"      带编/章/节层级的法律: {with_chap} / {len(meta)}")
    for name in ["中华人民共和国行政处罚法", "中华人民共和国民法典"]:
        segs = meta.get(name, {}).get("chapter_ranges", [])
        if segs:
            log(f"      {name} 章节区间示例: {segs[0]['path']} [{segs[0]['from']}-{segs[0]['to']}] ... 共 {len(segs)} 段")

    if args.dry_run:
        log("[dry-run] 不写文件，退出")
        return

    old = []
    if os.path.isdir(LAW_DATA_DIR):
        old = [f for f in os.listdir(LAW_DATA_DIR) if f.endswith(".txt")]
    if old and not args.keep_old:
        stamp = datetime.now().strftime("%Y%m%d")
        backup = os.path.join(BACKUP_ROOT, f"_backup_law_data_{stamp}")
        os.makedirs(backup, exist_ok=True)
        for f in old:
            shutil.move(os.path.join(LAW_DATA_DIR, f), os.path.join(backup, f))
        log(f"[3/5] 旧语料 {len(old)} 个文件已备份到 {os.path.relpath(backup, ROOT)}")
    else:
        log(f"[3/5] 旧语料 {len(old)} 个文件（keep-old 或目录为空，未备份）")

    os.makedirs(LAW_DATA_DIR, exist_ok=True)
    for title, articles in sorted(files.items()):
        with open(os.path.join(LAW_DATA_DIR, f"{title}.txt"), "w", encoding="utf-8", newline="\n") as fo:
            fo.write("\n".join(articles) + "\n")
    log(f"[4/5] 已写入 {len(files)} 部法律 -> {os.path.relpath(LAW_DATA_DIR, ROOT)}")

    # 收录语料目录里「不是本次生成」的文件（如 KuugoRen 补充的新法），保证元数据与语料一一对应
    adopted = adopt_unmanaged(LAW_DATA_DIR, files, meta)
    if adopted:
        log(f"      另收录 {len(adopted)} 部补充语料（非全量库快照，来源标注为 kuugoren）: {adopted}")

    with open(META_OUT, "w", encoding="utf-8") as fo:
        json.dump(meta, fo, ensure_ascii=False, indent=1)
    log(f"[5/5] 元数据（时效/章节目录）-> {os.path.relpath(META_OUT, ROOT)}")

    parsed = parse_law_dir(LAW_DATA_DIR)
    expect_total = sum(v["n_articles"] for v in meta.values())
    ok = len(parsed) == expect_total
    log(f"\n自检：文件 {len(os.listdir(LAW_DATA_DIR))} 个，解析出 {len(parsed)} 条"
        f"（元数据期望 {expect_total} 条）-> {'一致 OK' if ok else '不一致 FAIL'}")
    hist = [k for k, v in meta.items() if v.get("is_history")]
    log(f"      历史版本文件: {hist}")


if __name__ == "__main__":
    main()
