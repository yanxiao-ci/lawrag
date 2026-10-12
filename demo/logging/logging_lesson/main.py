from logger_base.logger import setup_logger

# 初始化日志记录器
logger = setup_logger("MainApp")

def process_data(data):
    logger.debug(f"开始处理数据: {data}")
    if not data:
        logger.error("数据为空，无法处理")
        return None
    logger.info("数据处理完成")
    return data.upper()

def main():
    logger.info("程序启动")
    result = process_data("hello")
    if result:
        logger.info(f"处理结果: {result}")
    else:
        logger.warning("处理失败")
    logger.info("程序结束")

if __name__ == "__main__":
    main()
