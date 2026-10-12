import logging
import os

# os.path.abspath(__file__) 获取当前文件的绝对路径
# os.path.dirname 获取参数对应的目录
log_dir = os.path.dirname(os.path.abspath(__file__))
print(log_dir)
log_file = os.path.join(log_dir, 'app.log')
print(f'log_file: {log_file}')

# 配置日志，输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,  # 日志文件路径    --输入文件里，不会输出到控制台
    encoding='utf-8',   # 指定字符集
    filemode='a'         # 'a'表示追加，'w'表示覆盖
)

# 获取日志记录器
logger = logging.getLogger("Example3")

# 记录日志
logger.debug('Debug')
logger.info("程序启动")
logger.warning("内存使用率较高")
logger.error("无法连接数据库")