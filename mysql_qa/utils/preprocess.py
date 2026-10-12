"""
    jieba分词工具类
        文本预处理的流程
            1.将英文统一转小写
            2.把句子进行分词

            什么时候调用？
            1.写入redis '分词后问题列表' 的时候（给bm25算法使用）
            2.用户提交query进行查询的时候
"""
# 导入分词库
import jieba
# 导入日志
from base.logger import logger


def preprocess_text(text):
    # 预处理文本
    logger.info(f"开始预处理文本-->{text}")
    try:
        # 转小写后再分词
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


if __name__ == '__main__':
    text = "AI大模型是什么？"
    print(preprocess_text(text))