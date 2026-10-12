"""
    集合索引其他操作
"""
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")
client.using_database(db_name="milvus_demo")


# 1.创建集合
def create_collection():
    # 定义schema 来描述集合和字段
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="", metric_type="COSINE", index_name="vector_index")
    # 创建集合，指定索引后会自动加载集合
    client.create_collection(collection_name="demo_v5", schema=schema, index_params=index_params)

# 2.集合操作
def operate_collection():
    #查询集合状态，未加载的集合不能添加数据
    print(client.get_load_state(collection_name="demo_v5"))
    # 加载集合：集合必须有索引
    client.load_collection(collection_name="demo_v5")
    # 查询集合状态
    print(client.get_load_state(collection_name="demo_v5"))
    # 释放集合
    client.release_collection(collection_name="demo_v5")
    print(client.get_load_state(collection_name="demo_v5"))
    # 删除索引
    client.drop_index(collection_name='demo_v5', index_name='vector_index')
    # 删除集合
    client.drop_collection(collection_name="demo_v5")
    print("集合删除成功")


# 测试
if __name__ == '__main__':
    create_collection()
    operate_collection()