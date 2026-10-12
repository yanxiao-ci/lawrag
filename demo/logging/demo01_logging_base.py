import logging

# 配置基本的日志设置，level表示日志级别，只有 >= 该级别的日志才会输出
logging.basicConfig(level=logging.DEBUG)

# 获取日志记录器
logger = logging.getLogger("Example1")
# 要看颜色，要装一个插件 Grep Console

# 记录不同级别的日志
logger.debug("这是调试信息，通常用于开发")
logger.info("程序运行正常")
logger.warning("注意，可能有小问题")
logger.error("发生错误")
logger.critical("严重错误，程序可能崩溃")