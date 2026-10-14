"""
    系统入口：新版   整合 FQA + RAG 的总入口
"""
# 导入 MySQL 系统组件
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入 RAG 系统组件
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.new_rag_system import RAGSystem
# 导入配置和日志
from base.law_parser import extract_citation
from base.logger import logger
from base.config import config
# 导入OpenAI客户端，用于DashScope API
from openai import OpenAI
# 导入时间库，记录处理时间
import time
# 导入UUID生成唯一会话ID
import uuid
# 导入pymysql错误处理
import pymysql


class IntegratedQASystem:
    def __init__(self):
        # 初始化MySQL客户端
        self.mysql_client = MySQLClient()
        # 初始化Redis客户端
        self.redis_client = RedisClient()
        # 初始化BM25搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            # 初始化OpenAI客户端，连接DashScope API
            self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                                 base_url=config.DASHSCOPE_BASE_URL)
        except Exception as e:
            logger.error(f"IntegratedQASystem OpenAI客户端初始化失败: {e}")
            raise
        # 初始化向量存储
        self.vector_store = VectorStore()
        # 初始化RAG系统
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        # 初始化对话历史表
        self.init_conversation_table()

    # 在MySQL中初始化对话历史表
    def init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        try:
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            self.mysql_client.connection.commit()
            logger.info("init_conversation_table 对话历史表初始化成功")
        except pymysql.MySQLError as e:
            logger.error(f"init_conversation_table 初始化对话历史表失败: {e}")
            raise

    # 调用模型，实现流式输出
    def call_dashscope(self, prompt):
        """调用DashScope API生成答案（流式输出）"""
        try:
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True  # 启用流式输出
            )

            # 收集流式输出的结果
            collected_content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    # 这里可以通过回调函数向前端发送每个chunk
                    yield content

            return collected_content
        except Exception as e:
            logger.error(f"call_dashscope LLM调用失败: {e}")
            # 注意：本函数是生成器，出错时**不能** `return "错误信息"`——
            # 生成器里 return 的值会被丢弃（只留在 StopIteration.value 里），
            # 上层"".join(...) 拿到的是空字符串，于是前端显示【空白回答】。
            # 2026-10-14 模型免费额度耗尽时就是这样被藏住 74 次失败的。
            # 改成 yield 出来，让错误对用户和日志都可见。
            reason = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
            yield (f"（模型调用失败：{reason}）\n"
                   f"抱歉，暂时无法生成回答，请稍后重试；若持续失败请检查模型额度配置，"
                   f"或联系人工客服：{config.CUSTOMER_SERVICE_PHONE}")

    # 从MySQL中获取最近的5轮会话历史
    def _fetch_recent_history(self, session_id: str) -> list:
        """获取最近5轮对话历史"""
        try:
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, 5))
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            logger.error(f"_fetch_recent_history 获取对话历史失败: {e}")
            return []

    # 更新会话历史
    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            history = self._fetch_recent_history(session_id)
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, 5))
            self.mysql_client.connection.commit()
            logger.info(f"update_session_history 会话 {session_id} 历史更新成功")
            return history
        except pymysql.MySQLError as e:
            logger.error(f"update_session_history 更新会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            logger.error(f"update_session_history 更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise

    # 根据session_id获取会话历史
    def get_session_history(self, session_id: str) -> list:
        """从MySQL获取会话历史"""
        return self._fetch_recent_history(session_id)

    # 根据session_id清除会话历史
    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        try:
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            self.mysql_client.connection.commit()
            logger.info(f"clear_session_history 会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            logger.error(f"clear_session_history 清除会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False

    # 小改动 B：条文号精确路由的辅助方法
    def _answer_by_citation(self, query):
        """小改动 B：条文号精确路由。
        查询里出现《X法》第Y条 这类精确引用时，直接走 MySQL 精确查询；
        确定性检索优先于语义检索——精确引用不该交给向量库去"猜"。
        P1：返回时带上 编/章/节 与施行日期，并提示是否另有历史版本。
        命中返回条文串，未命中（或没有引用）返回 None。
        """
        citation = extract_citation(query)
        if not citation:
            return None
        law_name, article_no, _num = citation
        detail = self.mysql_client.find_law_article_detail(law_name, article_no)
        if not detail:
            logger.info(f"条文号精确路由未命中: {law_name} {article_no}")
            return None
        logger.info(f"条文号精确命中: {detail['law_name']} {article_no} "
                    f"(章节: {detail.get('chapter') or '无'}, 状态: {detail.get('status')})")
        # 组装出处信息：编/章/节 + 施行日期（只有现行有效版本才标施行日）
        tags = []
        if detail.get("chapter"):
            tags.append(detail["chapter"])
        if detail.get("status") == "有效" and detail.get("effective_date"):
            tags.append(f"{detail['effective_date']} 施行")
        if detail.get("status") and detail["status"] != "有效":
            tags.append(f"版本状态：{detail['status']}")
        where = f"（{' · '.join(tags)}）" if tags else ""
        answer = f"《{detail['law_name']}》{article_no}{where}：{detail['article_text']}"
        # 时效提示：库里同时保留了历史版本，回答里明确说明引的是现行有效版本
        versions = self.mysql_client.list_law_versions(detail["law_name"])
        if versions:
            v = versions[0]
            answer += (f"\n\n（时效提示：以上引用的《{detail['law_name']}》为现行有效版本；"
                       f"知识库另存有 {str(v[2])[:4]} 年的旧版本，已标记为「{v[1]}」，"
                       f"如需新旧条文对照可以说一声。）")
        return answer

    # 问答查询
    def query(self, query, source_filter=None, session_id=None):
        """查询集成系统，支持对话历史和流式输出"""
        start_time = time.time()
        logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        # 获取会话历史

        #上下文token有限,所以限制五轮
        history = self.get_session_history(session_id) if session_id else []

        # 小改动 B：先试条文号精确路由（《X》第Y条 直接查 MySQL）
        answer = self._answer_by_citation(query)
        if answer is not None:
            need_rag = False
        else:
            # 其次尝试MySQL(BM25 法条正文 + 法律问答)精确查询
            answer, need_rag = self.bm25_search.search(query, threshold=0.85)

        if answer:
            # MySQL找到精确答案，直接返回
            logger.info(f"query MySQL答案: {answer}")
            if session_id:
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            logger.info(f"query 查询处理耗时 {processing_time:.2f}秒")
            # 对于MySQL答案，一次性返回
            yield answer, True  # 第二个参数表示这是完整答案
        elif need_rag:
            # 需要使用RAG系统，支持流式输出
            logger.info("query 无可靠MySQL答案，回退到RAG")
            # 收集完整答案以便存储
            collected_answer = ""
            # 从RAG系统获取流式输出
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                collected_answer += token
                yield token, False  # 第二个参数表示这是部分答案

            # 完整答案收集完毕，更新会话历史
            if session_id:
                self.update_session_history(session_id, query, collected_answer)

            processing_time = time.time() - start_time
            logger.info(f"query 查询处理耗时 {processing_time:.2f}秒")
            # 标记结束
            yield "", True  # 空字符串表示流结束，True表示完成
        else:
            # 未找到答案
            logger.info("query 未找到答案")
            processing_time = time.time() - start_time
            logger.info(f"query 查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True  # 一次性返回

def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的学科类别：{config.VALID_SOURCES}")
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break
            source_filter = input(f"请输入学科类别 ({'/'.join(config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in config.VALID_SOURCES:
                logger.warning(f"main 无效的学科类别 '{source_filter}'，将不过滤")
                source_filter = None

            print("\n答案: ", end="", flush=True)  # 开始打印答案
            answer = ""  # 用于累积完整答案
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:  # 仅当 token 非空时打印
                    print(token, end="", flush=True)  # 流式打印
                    answer += token  # 累积答案
                if is_complete:  # 如果是完整答案或流结束，换行并退出循环
                    print()  # 换行
                    break

            # 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        logger.error(f"main 系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()


if __name__ == "__main__":
    main()
