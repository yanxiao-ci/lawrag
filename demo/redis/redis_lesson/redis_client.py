import redis
import json
from redis_base.logger import logger

class RedisClient:
    def __init__(self):
        try:
            self.client = redis.StrictRedis(
                # host='localhost',
                host='192.168.229.128',
                port=6379,
                password='123456',
                db=0,   #使用0号库，默认是16个数据库，数据库 0-15
                decode_responses=True
            )
            logger.info(f"Redis 连接成功， client:{self.client}")
        except redis.RedisError as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

    # 存储键值对
    def set_data(self, key, value):
        try:
            # 使用json.dumps()将数据转换为JSON字符串
            self.client.set(key, json.dumps(value))
            logger.info(f"存储数据成功: {key}")
        except redis.RedisError as e:
            logger.error(f"Redis存储数据失败: {e}")

    def get_data(self, key):
        try:
            data = self.client.get(key)
            logger.info(f"获取数据成功: {key}")
            # 使用json.loads()将JSON字符串转换为Python对象
            return json.loads(data) if data else None
        except redis.RedisError as e:
            logger.error(f"获取数据失败: {e}")
            return None

if __name__ == "__main__":
    # 初始化 Redis 客户端
    redis_client = RedisClient()
    # 示例数据
    key = "user:1"
    value = {"name": "Alice", "age": 25}
    # # 存储数据
    redis_client.set_data(key, value)
    # # 获取数据
    result = redis_client.get_data(key)
    print(type(result))
    if result:
        logger.info(f"查询结果: {result}")
    else:
        logger.info("未找到数据")