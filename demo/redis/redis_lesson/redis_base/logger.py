import logging
import os

# 创建日志记录器
def setup_logger(name, log_file='logs/app.log'):
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置最低级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # 设置处理器格式
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器（避免重复添加）
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger("Redis")