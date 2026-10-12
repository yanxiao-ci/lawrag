"""
    MySQL客户端
"""
# 导入 MySQL 连接库
import pymysql
# 导入pandas
import pandas as pd
# 导入配置和日志
from base.config import config
from base.logger import logger


class MySQLClient:

    def __init__(self):
        try:
            # 初始化MySQL连接对象
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                port=config.MYSQL_PORT,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE
            )
            # 创建游标，用于执行SQL语句
            self.cursor = self.connection.cursor()
            # 记录连接成功
            logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            # 记录连接失败
            logger.error(f"MySQL 连接失败: {e}")
            raise

    # 创建表
    def create_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS jpkb (
                id INT AUTO_INCREMENT PRIMARY KEY,
                subject_name VARCHAR(20),
                question VARCHAR(1000),
                answer VARCHAR(1000))
            '''
        try:
            res = self.cursor.execute(create_table_query)
            logger.info(f"表创建成功:{res}")
        except pymysql.MySQLError as e:
            logger.error(f"表创建失败: {e}")
            raise

    # 插入指定csv文件的数据，只执行一次，如果要重复执行，先清空数据库
    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            logger.info(f"正在处理数据: {data}")
            logger.info(f"data: {type(data)}")
            for _, row in data.iterrows():
                logger.info(f"row: {row}")
                sql = 'INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)'
                res = self.cursor.execute(sql, (row['学科名称'], row['问题'], row['答案']))
            # 提交事务
            self.connection.commit()
            logger.info(f"数据插入成功:{res}")
        except Exception as e:
            logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise

    # 获取所有问题
    def fetch_questions(self):
        try:
            # 执行SQL
            self.cursor.execute('SELECT question FROM jpkb')
            # 获取结果
            res = self.cursor.fetchall()
            print(type(res))
            print(f'所有问题:{res}')
            # 其中一个元素：('用上下文管理器实现函数运行时间的计算?',)
            # 记录获取成功
            logger.info("成功获取问题")
            # 返回结果
            return res
        except pymysql.MySQLError as e:
            # 记录查询失败
            logger.error(f"查询失败: {e}")
            # 返回空列表
            return []

    # 获取指定问题的答案
    def fetch_answer(self, question):
        try:
            # 执行查询
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            # 获取结果，一条记录是一个tuple
            result = self.cursor.fetchone()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    # 关闭数据库连接
    def close(self):
        try:
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    mysql_client = MySQLClient()
    mysql_client.create_table()
    data_dir = r'..\data\JP学科知识问答.csv'
    mysql_client.insert_data(data_dir)
    mysql_client.fetch_questions()
    answer = mysql_client.fetch_answer('用上下文管理器实现函数运行时间的计算?')
    print(f'answer:{answer}')
    mysql_client.close()