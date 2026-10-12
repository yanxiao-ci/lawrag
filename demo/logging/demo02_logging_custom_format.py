import logging

# 配置日志格式
logging.basicConfig(
    level=logging.DEBUG,#可以自定义级别
    # 格式
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 获取日志记录器
logger = logging.getLogger("Example2")

# 记录日志
logger.debug("调试模式已开启")
logger.info("正在处理数据")
logger.error("数据处理失败")
