"""
    mysql_qa 检索     需要会写核心代码
"""
# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base.logger import logger

class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题分词后列表
        self.questions = None
        # 初始化原始问题列表
        self.original_questions = None
        # 加载数据
        self._load_data()

    # 加载MySQL中的数据，同步到Redis（原始问题、问题分词后列表），基于分词后列表初始化BM25模型
    def _load_data(self):
        # 定义Redis的key
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从Redis中加载数据
        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # 如果数据不存在，则从MySQL中加载数据
        if not self.original_questions or not tokenized_questions:
            # 从MySQL中查询所有问题
            self.original_questions = self.mysql_client.fetch_questions()
            # 判断原始问题列表是否为空
            if not self.original_questions:
                logger.warning("原始问题列表为空")
                return
            # self.original_questions 有值
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            self.redis_client.set_data(original_key, [q[0] for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)
        # 记录 BM25 初始化成功
        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        logger.info("BM25 模型初始化完成")

    # 归一化
    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    # 搜索查询
    # 返回值1 答案   返回值2 是否走RAG
    def search(self, query, threshold=0.85):
        if not query or not isinstance(query, str):
            # 记录无效查询
            logger.error("无效问题")
            # 返回 None 和 False
            return None, False
        try:
            # 从Redis根据问题检索答案
            cached_answer = self.redis_client.get_answer(query)
            # 有，直接返回
            if cached_answer:
                logger.info(f"从Redis获取答案: {query}")
                return cached_answer, False
            # 无，对问题分词
            query_tokens = preprocess_text(query)
            # BM25 计算得分
            scores = self.bm25.get_scores(query_tokens)
            # 归一化
            softmax_scores = self._softmax(scores)
            # 将 softmax_scores 中的元素求和
            # sum_scores = softmax_scores.sum()
            # print(f'sum_scores-->{sum_scores}')
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]
            print(f'best_idx-->{best_idx}, best_score-->{best_score}')
            # 判断 得分是否大于阈值
            if best_score > threshold:
                # 大于，查询数据库
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 有，写入Redis
                    logger.info(f"从数据库获取答案: {query}")
                    self.redis_client.set_data(f"answer:{query}", answer)
                    return answer, False
            # 返回 None 和 True
            logger.info(f"无法获取答案: {query}")
            return None, True
        except Exception as e:
            # 记录搜索失败
            logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            return None, True
