"""
    配置管理模块
"""
# 导入配置解析库
import configparser
# 导入路径操作库
import os
# 导入 .env 加载器（本地开发用；.env 已 gitignore，用于存放 API Key 等敏感配置）
from dotenv import load_dotenv

# 加载项目根目录下的 .env；文件不存在时静默跳过
# 注意：要在读取 os.getenv 之前执行，且 .env 的值会覆盖 config.ini
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=None):
        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 项目根目录
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
        # 日志目录
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        # 数据目录
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/data')
        # 模型目录
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/models')
        # 文档加载器目录（PDF/Word/PPT/图片 → 文本 + OCR）
        self.DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, 'rag_qa/document_loaders')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件 config.ini
        self.config.read(config_file, encoding='utf-8')

        # --> 从.env文件中获取配置，找不到则从config.ini中获取，再找不到使用 fallback备选值
        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='192.168.229.128'))
        # MySQL端口
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.getint('mysql', 'port', fallback=3307)))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='subjects_kg'))

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='192.168.229.128'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.getint('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.getint('redis', 'password', fallback='123456'))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=0)))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='192.168.229.128'))
        # Milvus 端口
        self.MILVUS_PORT = int(os.getenv('MILVUS_PORT', self.config.getint('milvus', 'port', fallback=19530)))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='itcast'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='lawrag'))

        # LLM 配置
        # LLM 模型名
        # self.LLM_MODEL = os.getenv('LLM_MODEL', self.config.get('llm', 'model', fallback='qwen3.7-max-preview'))
        self.LLM_MODEL = os.getenv('LLM_MODEL', self.config.get('llm', 'model', fallback='qwen3.8-max'))
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY',
                                           self.config.get('llm', 'dashscope_api_key',
                                                           # fallback='sk-your-key-here'))
                                                           fallback='sk-your-key-here'))
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL',
                                            self.config.get('llm', 'dashscope_base_url',
                                                            # fallback='https://dashscope.aliyuncs.com/compatible-mode/v1'))
                                                            fallback='https://ws-9yhii0u53he00riq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'))

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = int(os.getenv('PARENT_CHUNK_SIZE',
                                               self.config.getint('retrieval', 'parent_chunk_size', fallback=1000)))
        # 子块大小
        self.CHILD_CHUNK_SIZE = int(os.getenv('CHILD_CHUNK_SIZE',
                                              self.config.getint('retrieval', 'child_chunk_size', fallback=200)))
        # 父块重叠大小
        self.PARENT_CHUNK_OVERLAP = int(os.getenv('PARENT_CHUNK_OVERLAP',
                                                  self.config.getint('retrieval', 'parent_chunk_overlap',
                                                                     fallback=150)))
        # 子块重叠大小
        self.CHILD_CHUNK_OVERLAP = int(os.getenv('CHILD_CHUNK_OVERLAP',
                                                 self.config.getint('retrieval', 'child_chunk_overlap', fallback=30)))
        # 检索返回数量
        self.RETRIEVAL_K = int(os.getenv('RETRIEVAL_K',
                                         self.config.getint('retrieval', 'retrieval_k', fallback=5)))
        # 最终候选数量
        self.CANDIDATE_M = int(os.getenv('CANDIDATE_M',
                                         self.config.getint('retrieval', 'candidate_m', fallback=2)))

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            os.getenv('VALID_SOURCES',
                      self.config.get('app', 'valid_sources', fallback='["law"]')))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = os.getenv('CUSTOMER_SERVICE_PHONE',
                                                self.config.get('app', 'customer_service_phone',
                                                                fallback='13012345678'))

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')


config = Config()

if __name__ == '__main__':
    print(config.VALID_SOURCES)
    print(config.MYSQL_PORT)
    print(type(config.MYSQL_PORT))
