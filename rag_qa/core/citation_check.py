# -*- coding: utf-8 -*-
"""
引用可验证（小改动 C）
======================
法律场景最常见的幻觉是编造条文号（例如“根据《X法》第999条”）。
做法：把回答里的 《法律名》第Y条 全部抽出来，回查本轮检索到的上下文：
  - 全部命中：正常返回；
  - 有未命中：提示“未能在检索到的条文中找到依据”，由上层决定降级/重试。

为兼容全称/简称（中华人民共和国 + 数字写法），比对前统一：
  法律名去掉“中华人民共和国”前缀；条文号统一转成整数。
"""

import re

from base.law_parser import norm_law_name, cn2int

CITE_RE = re.compile(r"《([^》]+)》第([一二三四五六七八九十百千万零〇两0-9]+)条")


def extract_citations(text):
    """抽取并归一化文本中的 《法律名》第Y条，返回 set{(法律名, 条文号整型)}。"""
    out = set()
    for m in CITE_RE.finditer(text or ""):
        law = norm_law_name(m.group(1))
        num = cn2int(m.group(2))
        if law and num is not None:
            out.add((law, num))
    return out


def validate_citations(answer, context):
    """返回回答中引用了、但不在检索上下文里的 (法律名, 条文号) 列表。空列表表示全部命中。"""
    cited = extract_citations(answer)
    available = extract_citations(context)
    return sorted(cited - available)
