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

from base.config import config
# 导入现有的系统
from new_main import IntegratedQASystem

# 创建应用实例
app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")

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

# 定义日常问候用语模式和回复
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|Hi|hello|Hello)",
        "response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗|在不)",
        "response": "我在！我是黑马程序员，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"
    }
]


# 定义请求模型，前端给我的
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


# 定义响应模型，我给前端的
class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float


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
    print("222222222222222222")
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
    query_result = qa_system.query(request.query, request.source_filter, session_id)
    print(f"111--->query:{request.query}, source_filter:{request.source_filter}")
    
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
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
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


# 获取有效的学科类别
@app.get("/api/sources")
async def get_sources():
    return {"sources": config.VALID_SOURCES}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8080, reload=False)
