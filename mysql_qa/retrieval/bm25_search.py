"""
    mysql_qa 检索     法条 BM25 + 法律问答 FAQ
    改造要点（《法律方向改造方案书》§6.1）:
      检索源从 law_faq 的「问题」改为 law_articles 的「法条正文」——
      法律术语（表见代理、善意取得等）精确匹配比向量更强；
      法条未命中时，再用 law_faq 的法律问答做一层 FAQ 兜底。
    门控（2026-10-14 重标定）:
      命中判定不再用「全库 softmax 分 > 0.85」（语料扩容后过松，真实问题 90% 被抢答、约一半答错），
      改成词面判据：法条通道看「查询词覆盖率」，FAQ 通道看「词集合 Jaccard」。
      标定脚本 tools/tune_fast_channel_gate.py，参数 threshold 即两通道的门槛（默认 0.85）。
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
        # 法条：BM25 模型 + 元数据 (law_name, article_no, article_num, article_text)
        self.article_bm25 = None
        self.article_meta = None
        # 法律问答：BM25 模型 + 问题列表
        self.faq_bm25 = None
        self.faq_questions = None
        # 加载数据
        self._load_data()

    # 从 MySQL 加载语料，分别构建「法条 BM25」与「问答 BM25」
    def _load_data(self):
        # 主检索源：法条正文
        articles = self.mysql_client.fetch_law_articles()
        if articles:
            self.article_meta = articles
            self.article_bm25 = BM25Okapi([preprocess_text(a[3]) for a in articles])
            logger.info(f"法条 BM25 初始化完成，共 {len(articles)} 条")
        else:
            logger.warning("法条语料为空")
        # 兜底：法律问答（law_faq）
        self.faq_questions = self.mysql_client.fetch_questions()
        if self.faq_questions:
            self.faq_bm25 = BM25Okapi([preprocess_text(q[0]) for q in self.faq_questions])
            logger.info(f"法律问答 BM25 初始化完成，共 {len(self.faq_questions)} 条")

    # 归一化
    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    # 在给定 BM25 上取最高分文档，返回 (下标, 归一化分)
    def _best(self, bm25, tokens):
        if bm25 is None:
            return -1, 0.0
        softmax_scores = self._softmax(bm25.get_scores(tokens))
        idx = int(softmax_scores.argmax())
        return idx, float(softmax_scores[idx])

    # 在给定 BM25 上取最高分文档，返回 (下标, 原始分)
    # 门控改用原始分 + 词面判据，不再依赖 softmax：softmax 是对全库归一化的，
    # 语料从 97 部扩到 292 部后，同一个分值代表的东西完全变了。
    def _top1_raw(self, bm25, tokens):
        if bm25 is None:
            return -1, 0.0
        scores = bm25.get_scores(tokens)
        idx = int(scores.argmax())
        return idx, float(scores[idx])

    # 词覆盖率：查询词有多大比例出现在候选文本里（= 查询本身是不是就是这段文本）
    @staticmethod
    def _coverage(query_tokens, text):
        q = set(query_tokens)
        if not q:
            return 0.0
        return len(q & set(preprocess_text(text))) / len(q)

    # 词集合 Jaccard：两个问题算不算"同一个问题"
    @staticmethod
    def _jaccard(a, b):
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # 搜索查询
    # 返回值1 答案   返回值2 是否走RAG
    # 门控改动（2026-10-14）：从「全库 softmax 分 > 0.85」改成「词面判据」——
    #   旧逻辑是"像"就抢答：实测 100 条真实法律问题 90% 被快通道截获、约一半答错，RAG 形同虚设；
    #   新逻辑是"是"才抢答（同一个问题 / 查询本身就是那条条文），其余一律交给 RAG。
    #   阈值用 tools/tune_fast_channel_gate.py 标定（P=库内原句 200 条，N=真实法律问题 100 条）：
    #   库内原句 Jaccard 恒为 1.0，真实问题 P90 只有 0.5，0.85 这一档错答 0。
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

            # 1) 法条正文优先：只认「查询基本就是这条条文」（例如直接问某个法律术语）
            best_idx, raw_score = self._top1_raw(self.article_bm25, query_tokens)
            logger.info(f"法条 best_idx-->{best_idx}, raw-->{raw_score:.2f}")
            if best_idx >= 0:
                law_name, article_no, _num, article_text = self.article_meta[best_idx]
                cover = self._coverage(query_tokens, article_text)
                if cover >= threshold:
                    answer = f"《{law_name}》{article_no}：{article_text}"
                    logger.info(f"法条命中: {law_name} {article_no} (查询词覆盖率 {cover:.2f})")
                    self.redis_client.set_data(f"answer:{query}", answer)
                    return answer, False
                logger.info(f"法条未过门控: {law_name} {article_no} 查询词覆盖率 {cover:.2f} < {threshold}")

            # 2) 法律问答兜底：只认「同一个问题」（快问快答的本意）
            best_idx, raw_score = self._top1_raw(self.faq_bm25, query_tokens)
            logger.info(f"问答 best_idx-->{best_idx}, raw-->{raw_score:.2f}")
            if best_idx >= 0:
                original_question = self.faq_questions[best_idx][0]
                jac = self._jaccard(query_tokens, preprocess_text(original_question))
                if jac >= threshold:
                    answer = self.mysql_client.fetch_answer(original_question)
                    if answer:
                        logger.info(f"问答命中: {original_question} (Jaccard {jac:.2f})")
                        self.redis_client.set_data(f"answer:{query}", answer)
                        return answer, False
                else:
                    logger.info(f"问答未过门控: Jaccard {jac:.2f} < {threshold}")

            # 未命中，转 RAG
            logger.info(f"未命中，转 RAG: {query}")
            return None, True
        except Exception as e:
            # 记录搜索失败
            logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            return None, True
