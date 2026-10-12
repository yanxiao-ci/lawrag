"""
    BM25搜索
"""
import jieba
# from rank_bm25 import BM25L
from rank_bm25 import BM25Okapi
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BM25Search:
    def __init__(self, documents):
        # 初始化文档集合
        self.documents = documents
        # 分词后的文档
        self.tokenized_docs = [jieba.lcut(doc) for doc in documents]
        # 初始化BM25模型
        # self.bm25 = BM25L(self.tokenized_docs)
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info("BM25模型初始化完成")

    def search(self, query):
        # 分词查询
        tokenized_query = jieba.lcut(query)
        try:
            # 计算每个文档的BM25得分
            scores = self.bm25.get_scores(tokenized_query)
            logger.info(f'scores-->{scores}')
            # 获取最高得分的文档索引
            best_idx = scores.argmax()
            best_score = scores[best_idx]
            best_doc = self.documents[best_idx]
            logger.info(f"查询: {query}, 最佳匹配: {best_doc}, 得分: {best_score}")
            return best_doc, best_score
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return None, 0.0