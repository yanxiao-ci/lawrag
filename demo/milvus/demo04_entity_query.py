"""
    混合搜索
"""
import random
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker, RRFRanker

# 创建 Milvus 客户端
client = MilvusClient(uri="http://192.168.229.128:19530")
# 使用数据库 milvus_demo
client.using_database(db_name="milvus_demo")

# 1.创建集合并插入数据
def create_collection_and_data():
    # 定义schema
    schema = client.create_schema(enable_dynamic_field=False)
    # 创建字段
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="filmVector", datatype=DataType.FLOAT_VECTOR, dim=5)  # 向量字段
    schema.add_field(field_name="posterVector", datatype=DataType.FLOAT_VECTOR, dim=5)  # 向量字段
    # 给向量字段添加索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="filmVector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128},
                           index_name="film_vector_index")
    index_params.add_index(field_name="posterVector", index_type="", metric_type="COSINE",
                           index_name="poster_vector_index")
    # 创建集合
    client.create_collection(collection_name="demo_v3", schema=schema, index_params=index_params)

    # 批量插入数据
    entities = []
    for _ in range(1000):
        id = random.randint(0, 10000)
        film_vector = [random.random() for _ in range(5)]
        poster_vector = [random.random() for _ in range(5)]
        entity = {"id": id, "filmVector": film_vector, "posterVector": poster_vector}
        entities.append(entity)
    # 批量插入数据
    client.upsert(collection_name="demo_v3", data=entities)

# 2.混合搜索
def search():
    # 查询向量
    query_filmVector = [
        [0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]
    dense_search_params = {
        "data": query_filmVector,
        "anns_field": "filmVector",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 2
    }
    # 第一个搜索AnnSearchRequest
    request1 = AnnSearchRequest(**dense_search_params)

    query_posterVector = [
        [0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]
    sparse_search_params = {
        "data": query_posterVector,
        "anns_field": "posterVector",
        "param": {"metric_type": "COSINE"},
        "limit": 2
    }
    # 第二个搜索AnnSearchRequest
    request2 = AnnSearchRequest(**sparse_search_params)
    # 合并查询条件
    requests = [request1, request2]
    # 重排序
    # ranker = RRFRanker(100)
    ranker = WeightedRanker(0.3, 0.8)
    # 执行混合搜索
    results = client.hybrid_search(collection_name="demo_v3", reqs=requests, ranker=ranker, limit=2, output_fields=["id"])
    # print(results)
    for res in results:
        for r in res:
            print(r)
            # print(r['entity'].id)


if __name__ == '__main__':
    # create_collection_and_data()
    search()