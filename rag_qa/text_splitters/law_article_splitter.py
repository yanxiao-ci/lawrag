# -*- coding: utf-8 -*-
"""
法条边界感知切分器（小改动 A）
================================
通用递归切分器按字数切，会把法条从中间切断，检索到“半条法条”，引用就不准。
法条天然是自包含的最小单元，所以：
  child  块 = 一条完整法条（不跨条）        -> 保证引用精确
  parent 块 = 同一部法律里连续的若干条      -> 用于补齐上下文
父块用换行拼接若干条，child 再按行切回单条，因此 child 边界 == 法条边界。
"""

from langchain_core.documents import Document

from base.law_parser import parse_law_text, law_name_from_path


def standard_articles(text, law_name):
    """把整段法律文本切成标准化法条字符串：「《法律名》第X条 正文」。"""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for (ln, no, _num, body) in parse_law_text(line, law_name):
            out.append(f"《{ln}》{no} {body}")
    return out


class LawParentSplitter:
    """把一部法律按连续 group_size 条聚成一个父块。"""

    def __init__(self, group_size=5):
        self.group_size = group_size

    def split_documents(self, documents):
        parents = []
        for doc in documents:
            law_name = law_name_from_path(doc.metadata.get("file_path", ""))
            articles = standard_articles(doc.page_content, law_name)
            for start in range(0, len(articles), self.group_size):
                group = articles[start:start + self.group_size]
                parents.append(Document(page_content="\n".join(group), metadata=dict(doc.metadata)))
        return parents


class LawChildSplitter:
    """把父块（若干条）切成 child=单条法条。父块内每条独占一行，按行切即可。"""

    def split_documents(self, documents):
        children = []
        for doc in documents:
            for line in doc.page_content.splitlines():
                line = line.strip()
                if line:
                    children.append(Document(page_content=line, metadata=dict(doc.metadata)))
        return children
