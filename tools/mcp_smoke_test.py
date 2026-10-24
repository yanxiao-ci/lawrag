# -*- coding: utf-8 -*-
"""
MCP 冒烟测试：启动 mcp_server/server.py（stdio），逐个调用工具并打印结果。

前置：lawrag FastAPI 已启动（默认 http://127.0.0.1:8080）。
运行（用装了 mcp 的解释器，例如 edu_agent）：
  python tools/mcp_smoke_test.py
"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "mcp_server", "server.py")
BASE_URL = os.getenv("LAWRAG_BASE_URL", "http://127.0.0.1:8080")


def text_of(result):
    """把 CallToolResult 的 content 拍平成文本。"""
    out = []
    for item in result.content:
        out.append(getattr(item, "text", str(item)))
    return "\n".join(out)


async def main():
    env = {**os.environ, "LAWRAG_BASE_URL": BASE_URL,
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    params = StdioServerParameters(command=sys.executable, args=[SERVER], env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("工具列表：", [t.name for t in tools.tools], flush=True)

            cases = [
                ("get_corpus_info", {}),
                ("search_law", {"query": "试用期最长多久", "top_k": 3, "domain": "labor"}),
                ("get_article", {"law_name": "中华人民共和国妇女权益保障法", "article_no": "第43条"}),
                ("check_hr_compliance",
                 {"text": "招聘后端工程师，限男性，35岁以下，试用期9个月。", "scope": "jd"}),
            ]
            for name, args in cases:
                print(f"\n===== {name} =====", flush=True)
                result = await session.call_tool(name, arguments=args)
                print(text_of(result)[:2000], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
