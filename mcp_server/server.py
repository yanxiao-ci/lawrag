# -*- coding: utf-8 -*-
"""
lawrag MCP 薄封装
=================
只做协议适配：MCP 工具 -> lawrag 内部 HTTP 接口。
不 import app.py，不加载 BGE-M3 / Reranker / BERT，避免和 FastAPI 进程重复占 GPU。

工具（与方案契约一致）：
  - search_law       检索条文，只返回证据，不生成长文本
  - get_article      按法律名 + 条号取全文（支持 第四十三条 / 第43条 / 43 / 之一）
  - get_corpus_info  语料规模、领域、覆盖边界

传输：
  python mcp_server/server.py                 # stdio（开发 / Claude Code）
  python mcp_server/server.py streamable-http # HTTP（部署，默认 127.0.0.1:8000/mcp）

环境变量：
  LAWRAG_BASE_URL   默认 http://127.0.0.1:8080
  LAWRAG_TIMEOUT    默认 60 秒（检索要跑 GPU，留足时间）
"""
import os
import sys
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = os.getenv("LAWRAG_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
TIMEOUT = float(os.getenv("LAWRAG_TIMEOUT", "60"))

mcp = FastMCP(
    "lawrag",
    instructions=("中国法律法规检索服务。只返回可溯源的条文证据，不提供个案结论。"
                  "domain='labor' 时只检索 14 部劳动法与 HR 合规相关法律。"),
)


def _request(method: str, path: str, payload: Optional[dict] = None,
             timeout: float = TIMEOUT) -> dict:
    """调用 lawrag HTTP 接口；把连接/HTTP 错误转成可读的 RuntimeError。"""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, f"{BASE_URL}{path}", json=payload)
    except httpx.RequestError as e:
        raise RuntimeError(f"无法连接 lawrag（{BASE_URL}）：{e}") from e
    if resp.status_code >= 400:
        raise RuntimeError(f"lawrag {path} 返回 {resp.status_code}：{resp.text[:300]}")
    return resp.json()


@mcp.tool()
def search_law(query: str, top_k: int = 5, domain: Optional[str] = None,
               effective_only: bool = True) -> list:
    """检索法规条文，返回结构化证据；不生成长文本。

    Args:
        query: 检索问题，例如「试用期最长多久」「招聘限男性是否合法」
        top_k: 返回条数，1-20，默认 5
        domain: 领域名，如 "labor"（14 部劳动法）；不传则全量检索
        effective_only: 是否只要现行有效版本，默认 True
    """
    data = _request("POST", "/api/search_law", {
        "query": query,
        "top_k": top_k,
        "domain": domain,
        "effective_only": effective_only,
    })
    return data.get("results", [])


@mcp.tool()
def get_article(law_name: str, article_no: str) -> dict:
    """按法律名和条号取条文全文。

    Args:
        law_name: 法律名，如「中华人民共和国妇女权益保障法」
        article_no: 条号，支持「第四十三条」「第43条」「43」「第四十三条之一」
    """
    return _request("POST", "/api/get_article", {
        "law_name": law_name,
        "article_no": article_no,
    })


@mcp.tool()
def get_corpus_info() -> dict:
    """返回语料规模、可用领域和覆盖边界（给 Agent 说明能力范围）。"""
    return _request("GET", "/api/corpus_info")


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
