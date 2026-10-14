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
            CREATE TABLE IF NOT EXISTS law_faq (
                id INT AUTO_INCREMENT PRIMARY KEY,
                domain VARCHAR(20),
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
                sql = 'INSERT INTO law_faq (domain, question, answer) VALUES (%s, %s, %s)'
                res = self.cursor.execute(sql, (row['法律领域'], row['问题'], row['答案']))
            # 提交事务
            self.connection.commit()
            logger.info(f"数据插入成功:{res}")
        except Exception as e:
            logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise

    # 创建法条表（项目方案书 §4.1）
    def create_law_articles_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS law_articles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                law_name VARCHAR(100) NOT NULL COMMENT '法律名',
                article_no VARCHAR(32) NOT NULL COMMENT '条文号原文，如 第一千零六十二条',
                article_num INT NOT NULL COMMENT '条文号数值，用于排序与范围查询',
                article_text TEXT NOT NULL COMMENT '条文正文',
                chapter VARCHAR(200) NOT NULL DEFAULT '' COMMENT '编/章/节，如 第五编 婚姻家庭 第三章 家庭关系',
                publish_date VARCHAR(12) NOT NULL DEFAULT '' COMMENT '公布日期',
                effective_date VARCHAR(12) NOT NULL DEFAULT '' COMMENT '施行日期',
                status VARCHAR(12) NOT NULL DEFAULT '有效' COMMENT '有效/已修改/已废止（时效过滤依据）',
                UNIQUE KEY uk_law_article (law_name, article_no),
                KEY idx_law_num (law_name, article_num),
                KEY idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''
        try:
            res = self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info(f"法条表创建成功:{res}")
            self._migrate_law_articles()
        except pymysql.MySQLError as e:
            logger.error(f"法条表创建失败: {e}")
            raise

    # 增量迁移：老库里的 law_articles 没有 P1 的时效/层级字段，这里补齐（幂等，可反复执行）
    def _migrate_law_articles(self):
        self.cursor.execute(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'law_articles'")
        have = {r[0] for r in self.cursor.fetchall()}
        add_cols = {
            "chapter": "VARCHAR(200) NOT NULL DEFAULT '' COMMENT '编/章/节'",
            "publish_date": "VARCHAR(12) NOT NULL DEFAULT '' COMMENT '公布日期'",
            "effective_date": "VARCHAR(12) NOT NULL DEFAULT '' COMMENT '施行日期'",
            "status": "VARCHAR(12) NOT NULL DEFAULT '有效' COMMENT '有效/已修改/已废止'",
        }
        for col, ddl in add_cols.items():
            if col not in have:
                self.cursor.execute(f"ALTER TABLE law_articles ADD COLUMN {col} {ddl}")
                logger.info(f"law_articles 补字段: {col}")
        self.cursor.execute(
            "SELECT INDEX_NAME FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'law_articles'")
        if "idx_status" not in {r[0] for r in self.cursor.fetchall()}:
            self.cursor.execute("ALTER TABLE law_articles ADD INDEX idx_status (status)")
            logger.info("law_articles 补索引: idx_status")
        self.connection.commit()

    # 批量写入法条
    # rows: [(law_name, article_no, article_num, article_text, chapter, publish_date, effective_date, status), ...]
    def insert_law_articles(self, rows):
        sql = ('INSERT IGNORE INTO law_articles '
               '(law_name, article_no, article_num, article_text, chapter, publish_date, effective_date, status) '
               'VALUES (%s, %s, %s, %s, %s, %s, %s, %s)')
        try:
            res = self.cursor.executemany(sql, rows)
            self.connection.commit()
            logger.info(f"法条写入成功:{res}")
            return res
        except Exception as e:
            logger.error(f"法条写入失败: {e}")
            self.connection.rollback()
            raise

    # 批量导入 law_faq（法律问答），不逐行打日志
    def insert_faq_csv(self, csv_path):
        sql = 'INSERT INTO law_faq (domain, question, answer) VALUES (%s, %s, %s)'
        try:
            data = pd.read_csv(csv_path)
            rows = list(zip(data['法律领域'], data['问题'], data['答案']))
            res = self.cursor.executemany(sql, rows)
            self.connection.commit()
            logger.info(f"law_faq 导入成功:{res}")
            return res
        except Exception as e:
            logger.error(f"law_faq 导入失败: {e}")
            self.connection.rollback()
            raise

    # 取全部现行有效法条（BM25 检索用；历史版本不参与检索，只留作对照）
    def fetch_law_articles(self):
        try:
            self.cursor.execute(
                "SELECT law_name, article_no, article_num, article_text FROM law_articles "
                "WHERE status = '有效'")
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            logger.error(f"法条查询失败: {e}")
            return []

    # 按 法律名+条文号 精确查条文（§5.2 条文号精确路由用）；兼容全称/简称
    def find_law_article(self, law_name, article_no):
        try:
            self.cursor.execute(
                "SELECT article_text FROM law_articles "
                "WHERE REPLACE(law_name, '中华人民共和国', '')=%s AND article_no=%s",
                ((law_name or '').replace('中华人民共和国', ''), article_no))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except pymysql.MySQLError as e:
            logger.error(f"法条精确查询失败: {e}")
            return None

    # P1：条文号精确路由的完整结果（带 编/章/节 与时效），优先返回现行有效版本
    def find_law_article_detail(self, law_name, article_no):
        try:
            self.cursor.execute(
                "SELECT law_name, article_no, article_text, chapter, publish_date, effective_date, status "
                "FROM law_articles "
                "WHERE REPLACE(law_name, '中华人民共和国', '')=%s AND article_no=%s "
                "ORDER BY (status='有效') DESC, publish_date DESC LIMIT 1",
                ((law_name or '').replace('中华人民共和国', ''), article_no))
            row = self.cursor.fetchone()
            if not row:
                return None
            return {"law_name": row[0], "article_no": row[1], "article_text": row[2], "chapter": row[3],
                    "publish_date": row[4], "effective_date": row[5], "status": row[6]}
        except pymysql.MySQLError as e:
            logger.error(f"法条精确查询(详细)失败: {e}")
            return None

    # P1：查某部法律在库里的历史版本（用于「该条另有历史版本」的提示）
    def list_law_versions(self, law_name):
        base = (law_name or '').replace('中华人民共和国', '')
        try:
            self.cursor.execute(
                "SELECT law_name, status, publish_date, effective_date, COUNT(*) FROM law_articles "
                "WHERE REPLACE(law_name, '中华人民共和国', '') LIKE %s "
                "AND status <> '有效' GROUP BY law_name, status, publish_date, effective_date "
                "ORDER BY publish_date DESC",
                (base + "（%",))
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            logger.error(f"历史版本查询失败: {e}")
            return []

    # 获取所有问题
    def fetch_questions(self):
        try:
            # 执行SQL
            self.cursor.execute('SELECT question FROM law_faq')
            # 获取结果
            res = self.cursor.fetchall()
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
            self.cursor.execute("SELECT answer FROM law_faq WHERE question=%s", (question,))
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
    data_dir = r'..\data\law_qa.csv'
    mysql_client.insert_data(data_dir)
    mysql_client.fetch_questions()
    answer = mysql_client.fetch_answer('租房合同到期，房东不退押金怎么办？')
    print(f'answer:{answer}')
    mysql_client.close()
