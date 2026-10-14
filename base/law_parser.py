# -*- coding: utf-8 -*-
"""
法条解析工具（公共层）
把 rag_qa/data/law_data 里的法律 .txt 解析成结构化法条，供 MySQL 入库、法条切分、条文号精确路由共用。
语料有两种排版（对齐《法律方向改造方案书》§1.1）:
  主规则  : 《法律名》第X条规定，内容   (覆盖约 96.6%)
  兜底规则: 第X条 内容                  (法律名从文件名推断)
"""

import os
import re

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

_NUM = r"[一二三四五六七八九十百千万零〇两0-9]+"
# 「第X条之一 / 之二」是修正案新增条文（刑法共 53 条），必须并入条文号，
# 否则会和主条号重号，撞 law_articles 的 UNIQUE KEY(law_name, article_no)。
_SUF = r"(?:之(?P<suf>[一二三四五六七八九十]{1,3}))?"
MAIN_RE = re.compile(r"^《(?P<law>[^》]+)》第(?P<num>" + _NUM + r")条" + _SUF + r"规定，(?P<text>.*)$")
FALLBACK_RE = re.compile(r"^第(?P<num>" + _NUM + r")条" + _SUF + r"[\s　]*(?P<text>.*)$")
QUERY_RE = re.compile(r"《(?P<law>[^》]+)》第(?P<num>" + _NUM + r")条" + _SUF)


def article_no_of(m):
    """由正则匹配对象拼条文号：第X条 或 第X条之N。"""
    suf = m.group("suf") if "suf" in m.groupdict() else None
    return f"第{m.group('num')}条" + (f"之{suf}" if suf else "")


def cn2int(s):
    """中文数字转整数；无法解析返回 None。支持 一/十/百/千/万 及阿拉伯数字。"""
    if s is None:
        return None
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


def norm_law_name(name):
    """统一法律名写法（去掉'中华人民共和国'前缀），用于全称/简称比对。"""
    return (name or "").replace("中华人民共和国", "").strip()


def law_name_from_path(path):
    """从文件路径推断法律名（文件名去掉扩展名）。"""
    return os.path.splitext(os.path.basename(path or ""))[0]


def parse_law_text(text, law_name):
    """解析单部法律的整段文本，返回 [(law_name, article_no, article_num, article_text), ...]"""
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = MAIN_RE.match(line)
        if m is None:
            m = FALLBACK_RE.match(line)
        if m is None:
            continue
        article_no = article_no_of(m)
        article_num = cn2int(m.group("num"))
        if article_num is None:
            continue
        rows.append((law_name, article_no, article_num, m.group("text").strip()))
    return rows


def parse_law_file(path):
    """解析单个 .txt 法律文件，法律名取文件名（不含扩展名）。"""
    law_name = law_name_from_path(path)
    with open(path, encoding="utf-8") as f:
        return parse_law_text(f.read(), law_name)


def parse_law_dir(directory):
    """解析目录下所有 .txt 法律文件。"""
    rows = []
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith(".txt"):
            rows.extend(parse_law_file(os.path.join(directory, fn)))
    return rows


def extract_citation(query):
    """从用户查询里抽取 《X》第Y条 引用。命中返回 (law_name原文, article_no, article_num)，否则 None。"""
    if not query:
        return None
    m = QUERY_RE.search(query)
    if not m:
        return None
    num = m.group("num")
    v = cn2int(num)
    if v is None:
        return None
    return m.group("law"), article_no_of(m), v


# ---------------- 语料元数据（时效 / 编章节）P1 ----------------

def load_law_meta(path):
    """读取 law_meta.json（语料元数据：生效状态 + 编章节区间）。文件缺失时返回 {}。"""
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def chapter_of(meta, law_name, article_num):
    """按 法律名 + 条文号数值 查所属 编/章/节；查不到返回空串。"""
    info = (meta or {}).get(law_name) or {}
    for seg in info.get("chapter_ranges", []):
        if seg.get("from") is not None and seg["from"] <= article_num <= seg["to"]:
            return seg.get("path", "")
    return ""
