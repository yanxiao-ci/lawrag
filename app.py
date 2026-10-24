"""
    FastAPI接口    FastAPI 接口 + 前端页面
"""
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
import os
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional
import time
import re
import threading
from datetime import date

from base.config import config
from base.domains import list_domains, law_names_for
from mysql_qa.db.mysql_client import MySQLClient
# 导入现有的系统
from new_main import IntegratedQASystem

# 创建应用实例
app = FastAPI(title="法律法规智能问答API", description="法律法规智能问答助手（法条检索 + 法律问答 + RAG）")

# 配置CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
os.makedirs("static", exist_ok=True)

# 创建全局QA系统实例
qa_system = IntegratedQASystem()

# 工具接口专用：独立 DB 连接 + 锁，避免和对话链路的共享游标互相踩
tool_db = MySQLClient()
tool_db_lock = threading.Lock()
tool_search_lock = threading.Lock()

# 定义日常问候用语模式和回复（法律问答场景）
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|Hi|hello|Hello)",
        "response": "你好！我是法律法规智能问答助手，可以帮你查法条、答法律问题，很高兴为你服务。"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you|你能做什么|你会什么)",
        "response": "我是法律法规智能问答助手，覆盖 288 部现行法律（21,312 条法条）和 2 万条法律问答。你可以直接问「《民法典》第一千零六十二条讲什么」，也可以问「买到假货怎么赔」这类实际问题，回答会标注《法律名》第 X 条。"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗|在不)",
        "response": "我在！我是法律法规智能问答助手，随时为你解答法律问题。"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答法律法规相关的问题。有什么我可以帮你的？"
    }
]


# 定义请求模型，前端给我的
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None
    # 领域视图：domain 为领域名（如 labor），domain_mode 为 off / soft / strict
    domain: Optional[str] = None
    domain_mode: Optional[str] = "off"
    allow_faq: Optional[bool] = True


# 定义响应模型，我给前端的
class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float


# MCP / Agent 工具接口的请求模型（只检索，不生成长文本）
class SearchLawRequest(BaseModel):
    query: str
    top_k: int = 5
    domain: Optional[str] = None
    effective_only: bool = True


class GetArticleRequest(BaseModel):
    law_name: str
    article_no: str


# 添加静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


# 根路径重定向到index.html
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


# 创建新会话
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# 查询历史消息
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = qa_system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 清除历史消息
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None

"""
    API接口四要素：
        请求URL：/api/query
        请求方式：post
        请求参数：
            {
                "query": "pycharm专业版如何激活",
                "source_filter": "ai",
                "session_id": "a7ccd3a0-ad4b-46eb-857c-449a5d119cff"
            }  
        响应数据：
            {
                "answer": "第一种方式：\n\n这种方式是在线激活，也就是说必须要联网，在激活后使用的过程中，也必须联网。。。\n\n不要把pycharm升级到最新版，如果你升级了，第一种方式可能就不能激活了。。。\n\n1.  打开pycharm。打开下面的选项：\n\n2. 在输入框中输入：http://idea.imsxm.com  然后点击 Activate即可。注意：这个网址，如果失效了，从网上重新搜索一个。\n\n第二种方式：\n\n这种方式属于离线方式，激活的时候需要联网，激活之后，正常使用的过程，不需要联网。\n\n1.  修改 hosts文件。windows和ubuntu是一样的，只不过文件存在位置不同。\n\nsudo vi /etc/hosts ,打开文件，在文件最后添加：0.0.0.0 account.jetbrains.com\n\n2. 使用 注册码 激活。注意：注册码是有时间限制的，网上再找一个激活码。\n\n把 激活码 复制进去，点击 OK。",
                "is_streaming": false,
                "session_id": "a7ccd3a0-ad4b-46eb-857c-449a5d119cff",
                "processing_time": 0.02270221710205078
            }          
"""
# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 判断是否需要流式处理（基于 need_rag）
    query_result = qa_system.query(
        request.query, request.source_filter, session_id,
        domain=request.domain, domain_mode=request.domain_mode or "off",
        allow_faq=request.allow_faq if request.allow_faq is not None else True)
    
    first_chunk = next(query_result, None)
    if first_chunk is None:
        return {
            "answer": "未找到答案",
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    
    answer, is_complete = first_chunk
    
    if is_complete:
        return {
            "answer": answer,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    else:
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }


# 流式查询WebSocket接口
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            request_data = json.loads(data)

            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            domain = request_data.get("domain")
            domain_mode = request_data.get("domain_mode") or "off"
            allow_faq = request_data.get("allow_faq", True)

            start_time = time.time()

            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })

            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break

            # 调用QA系统进行查询，流式返回结果
            collected_answer = ""
            for token, is_complete in qa_system.query(
                    query, source_filter=source_filter, session_id=session_id,
                    domain=domain, domain_mode=domain_mode, allow_faq=allow_faq):
                collected_answer += token

                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break

                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })

                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break

                await asyncio.sleep(0.01)

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 获取可用的知识库来源（当前只有 law）
@app.get("/api/sources")
async def get_sources():
    return {"sources": config.VALID_SOURCES}


# 获取可用的领域视图（劳动法等）
@app.get("/api/domains")
async def get_domains():
    return {"domains": list_domains()}


# 检索-only：给 MCP / Agent 用，只返回证据，不生成长文本
@app.post("/api/search_law")
def search_law(request: SearchLawRequest):
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    top_k = max(1, min(int(request.top_k or 5), 20))

    law_names = None
    if request.domain:
        law_names = law_names_for(request.domain)
        if not law_names:
            raise HTTPException(status_code=404, detail=f"未知领域: {request.domain}")

    try:
        with tool_search_lock:
            docs = qa_system.vector_store.search_children(
                query, k=top_k, source_filter="law",
                effective_only=bool(request.effective_only),
                law_names=law_names, with_score=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")

    results = [{
        "law_name": d.metadata.get("law_name", ""),
        "article_no": d.metadata.get("article_no", ""),
        "text": d.page_content,
        "chapter": d.metadata.get("chapter", ""),
        "status": d.metadata.get("status", ""),
        "score": d.metadata.get("score"),
    } for d in docs]
    return {
        "query": query,
        "domain": request.domain,
        "effective_only": bool(request.effective_only),
        "count": len(results),
        "results": results,
    }


# 取条：按法律名 + 条号取全文，条号支持 第四十三条 / 第43条 / 43 / 第四十三条之一
@app.post("/api/get_article")
def get_article(request: GetArticleRequest):
    law_name = (request.law_name or "").strip()
    article_no = (request.article_no or "").strip()
    if not law_name or not article_no:
        raise HTTPException(status_code=400, detail="law_name 和 article_no 不能为空")
    with tool_db_lock:
        detail = tool_db.find_law_article_detail_flexible(law_name, article_no)
    if not detail:
        raise HTTPException(status_code=404, detail=f"未找到《{law_name}》{article_no}")
    return detail


# 语料信息：给 Agent 说明覆盖范围和规模
@app.get("/api/corpus_info")
def get_corpus_info():
    with tool_db_lock:
        tool_db.cursor.execute(
            "SELECT COUNT(*), COUNT(DISTINCT law_name), SUM(status='有效') FROM law_articles")
        total, laws, valid = tool_db.cursor.fetchone()
    return {
        "law_count": int(laws or 0),
        "article_count": int(total or 0),
        "valid_article_count": int(valid or 0),
        "as_of": date.today().isoformat(),
        "domains": list_domains(),
        "coverage_note": "仅覆盖全国人大及其常委会制定的法律级文本；不含行政法规、司法解释、地方规定。",
    }


if __name__ == "__main__":
    import uvicorn

    # host=0.0.0.0 表示"监听本机所有网卡"，它不是可访问的网址；
    # 浏览器请用 http://127.0.0.1:8080/ 或 http://localhost:8080/
    print("\n[提示] 服务启动后，请在浏览器打开: http://127.0.0.1:8080/")
    print("[提示] 不要用 http://0.0.0.0:8080/ —— 0.0.0.0 是监听地址，不是可访问地址\n")
    uvicorn.run(app=app, host="0.0.0.0", port=8080, reload=False)
