"""
    目的：创建向量数据库
"""
from pymilvus import MilvusClient


# 定义函数
def operate_db():
    # 1. 创建连接
    # client = MilvusClient(uri="http://localhost:19530")
    # 注意：IP 换成你刚才查到的虚拟机 IP，不要加空格
    client = MilvusClient(uri="http://192.168.229.128:19530")#这是我的虚拟机地址
    # 2. 获取数据库列表
    databases = client.list_databases()
    print(f"数据库列表：{databases}")
    # 3. 判断是否存在要创建的数据库 milvus_demo
    # 3.1 不存在，创建
    if "milvus_demo" not in databases:
        client.create_database(db_name="milvus_demo")
        print(f"创建数据库milvus_demo成功")
    else:
    # 3.2 存在，使用
        print(f"数据库milvus_demo已存在")
        client.using_database(db_name="milvus_demo")
    return client


# 测试
if __name__ == '__main__':
    operate_db()