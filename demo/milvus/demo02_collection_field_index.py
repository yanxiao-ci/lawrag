"""
    目的：创建collection和index
"""
from pymilvus import MilvusClient, DataType


# def operate_collection():
#     # 1. 建立连接，使用数据库
#     client = MilvusClient(uri="http://192.168.229.128:19530")
#     client.using_database(db_name="milvus_demo")
#     # 2. 指定字段
#     schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
#     schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
#     schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=10, description="向量字段")
#     schema.add_field(field_name="scalar1", datatype=DataType.VARCHAR, max_length=100, description="标题字段")
#     # 3. 创建集合
#     client.create_collection(collection_name="demo_v1", schema=schema)
#     # 4. 创建向量索引
#     index_params = client.prepare_index_params()
#     index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", index_name="vector_index", params={ "nlist": 128 })
#     client.create_index(collection_name="demo_v1", index_params=index_params)
#     # 5. 查询标量索引
#     lists = client.list_indexes(collection_name="demo_v1")
#     print(f"索引列表：{lists}")
#     res = client.describe_index(collection_name="demo_v1", index_name="vector_index")
#     print(f'vector_index:{res}')
#     # 6. 创建标量索引
#     index_params1 = client.prepare_index_params()
#     index_params1.add_index(field_name="scalar1", index_type="", index_name="scalar1_index")
#     client.create_index(collection_name="demo_v1", index_params=index_params1)
#     # 5. 查询标量索引
#     lists = client.list_indexes(collection_name="demo_v1")
#     print(f"索引列表：{lists}")
#     res = client.describe_index(collection_name="demo_v1", index_name="scalar1_index")
#     print(f'scalar1_index:{res}')


def operate_collection():
    #1.创建连接和数据库
    client = MilvusClient(uri="http://192.168.229.128:19530")
    # client.create_database(db_name="milvus_demo")
    client.using_database(db_name="milvus_demo")
    #2.创建集合
    #2.1.创建约束，字段
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    #3.2.增加字段
    schema.add_field(field_name="id", datatype=DataType. INT64, is_primary=True)
    #dim维度，只有向量字段有
    schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="my_text", datatype=DataType.VARCHAR, max_length=512)

    #3.3. 增加索引
    index_params=client.prepare_index_params()
    #创建主键索引
    index_params.add_index(field_name="id",index_params="STL_SORT",index_name="id",params={})
    # 创建向量索引。当使用index_type="AUTOINDEX"的时候，不能指定metric_type和params参数
    index_params.add_index(
        field_name="my_vector",
        #index_type="AUTOINDEX",
        index_type="IVF_FLAT",
        index_name="idx_vector_my_vector",
        metric_type="IP",
        params={
            "nlist": 64,  #指定簇数量
        }
    )

    client.create_collection(collection_name="demo_v1", schema=schema,index_params=index_params)
    print("demo_v1集合创建成功")


    #3.删除集合
    # client.drop_collection(collection_name="demo_v1")

def operate_index():
    client = MilvusClient(uri="http://192.168.229.128:19530")
    client.using_database(db_name="milvus_demo")
    indices = client.list_indexes (collection_name="demo_v1")
    print(f"indices--->{indices}")
    index_desc = client.describe_index(collection_name="demo_v1", index_name="idx_vector_my_vector")
    print(f"index_desc--->{index_desc}")
    #先释放集合，再删除索引时
    client.release_collection(collection_name="demo_v1")
    client.drop_index(collection_name="demo_v1", index_name="idx_vector_my_vector")

# 测试
if __name__ == '__main__':
    operate_collection()
    operate_index()