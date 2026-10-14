# cache/redis_client.py
# 导入 Redis 客户端
import redis
# 导入 JSON 处理
import json
# 导入配置和日志
from base.config import config
from base.logger import logger


class RedisClient:
    # 答案缓存的过期时间（秒）：7 天。
    # P1 收尾：原来 set 不带过期，命中的答案（含被 BM25 误命中的错误答案）会永久留在 Redis 里，
    # 法条更新后也会一直命中旧答案。7 天是个保守值，要调就改这一行（或以后挪进 config.ini）。
    ANSWER_TTL = 7 * 24 * 3600

    def __init__(self):
        try:
            # 连接 Redis
            self.client = redis.StrictRedis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )
            # 记录连接成功
            logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 记录连接失败
            logger.error(f"Redis 连接失败: {e}")
            raise

    def set_data(self, key, value, expire=ANSWER_TTL):
        # 存储数据到 Redis（expire 单位秒；传 None 表示永不过期）
        try:
            # 存储 JSON 数据    key:value   key是string， value是json串
            # ensure_ascii=False 加上中文就能正常显示
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=expire)
            # 记录存储成功
            logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            # 记录存储失败
            logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        # 从 Redis 获取数据
        try:
            # 获取数据
            data = self.client.get(key)
            # 返回解析后的 JSON 数据或 None
            return json.loads(data) if data else None
        except redis.RedisError as e:
            # 记录获取失败
            logger.error(f"Redis 获取失败: {e}")
            # 返回 None
            return None

    def get_answer(self, query):
        # 获取查询的缓存答案
        try:
            # 从 Redis 获取答案
            answer = self.get_data(f"answer:{query}")
            if answer:
                # 记录获取成功
                logger.info(f"从 Redis 获取答案: {query}")
                # 返回答案
                return answer
            # 返回 None
            return None
        except redis.RedisError as e:
            # 记录查询失败
            logger.error(f"Redis 查询失败: {e}")
            # 返回 None
            return None


if __name__ == '__main__':
    redis_client = RedisClient()
    # redis_client.set_data("AI大模型是什么？", "AI大模型是一种基于人工智能算法的计算机模型，用于处理大量数据，并生成具有挑战性的预测结果。")
    # result = redis_client.get_data("AI大模型是什么？")
    # print(result)
#     result = """
#         举个例子什么是实例
#
# 人类，是人的类，不是一个具体，但是小明，就是一个具体的实例
#
# 具体到代码上
#
# class Person():
#
#
#       def test1(self):
#
#              pass
#
# 小明=Person()
#
# 小明.test1()
#
# 小明就是实例，通过小明调用的方法，都是实例方法，也就是带self的
#
# 比如test1就是实例方法，是必须通过实例调用才可以的，而不可以通过Person.test()这种方式
#     """
    # 将来从MySQL能够检索到答案，通过下面的API写入Redis
    redis_client.set_data("answer:实例变量/实例方法怎么理解", "国庆过后再说")
    # 从Redis中通过问题检索答案
    res = redis_client.get_answer("实例变量/实例方法怎么理解")
    print(res)
