"""
    实体（数据）操作
"""
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://192.168.229.128:19530")
client.using_database(db_name="milvus_demo")


# 1.创建集合
def create_collection():
    # 定义schema 来描述集合和字段
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    # 索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="", metric_type="COSINE", index_name="vector_index")
    # 创建集合，指定索引后会自动加载集合
    client.create_collection(collection_name="demo_v2", schema=schema, index_params=index_params)


# 2.增加实体、多次执行插入重复的数据，会增加实体，但是数据不变
def insert_entity():
    data = [
        {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                             0.9029438446296592], "color": "pink_8682"},
        {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501,
                             0.838729485096104], "color": "red_7025"},
        {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185,
                             0.20785793220625592], "color": "orange_6781"},
        {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995,
                             0.95791889146345], "color": "pink_9298"},
        {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184,
                             0.30337481143159106], "color": "red_4794"},
        {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383,
                             -0.1446277761879955], "color": "yellow_4222"},
        {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192,
                             -0.8984947637863987], "color": "red_9392"},
        {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709,
                             0.5378064918413052], "color": "grey_8510"},
        {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872,
                             -0.6140360785406336], "color": "white_9381"},
        {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717,
                             -0.6980531615588608], "color": "purple_4976"}
    ]
    res = client.insert(collection_name="demo_v2", data=data)
    print(f"插入数据：{res}")


# 2.增加或更新实体、多次执行会自动判断实体是否存在，不存在增加，存在则更新
def upsert_entity():
    data = [
        {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
                              0.6699215060604258], "color": "red_1202"},
        {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
                              -0.14152860714878068], "color": "blue_4150"},
        {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
                              -0.8351194181285713], "color": "orange_4590"},
        {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
                              -0.0010987671273892108], "color": "pink_9619"},
        {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.8230244263466688,
                              -0.7986720938400362], "color": "orange_4863"},
        {"id": 15, "vector": [-0.8825129181091456, -0.9204557711667729, -0.935350065513425, 0.5484069690287079,
                              0.24448151140671204], "color": "orange_7984"},
        {"id": 16, "vector": [0.6285586391568163, 0.5389064528263487, -0.3163366239905099, 0.22036279378888013,
                              0.15077052220816167], "color": "blue_9010"},
        {"id": 17, "vector": [-0.20151825016059233, -0.905239387635804, 0.6749305353372479, -0.7324272081377843,
                              -0.33007998971889263], "color": "blue_4521"},
        {"id": 18, "vector": [0.2432286610792349, 0.01785636564206139, -0.651356982731391, -0.35848148851027895,
                              -0.7387383128324057], "color": "orange_2529"},
        {"id": 19, "vector": [0.055512329053363674, 0.7100266349039421, 0.4956956543575197, 0.24541352586717702,
                              0.4209030729923515], "color": "red_9437"}
    ]

    # 以后都用更新插入数据
    res = client.upsert(collection_name="demo_v2", data=data)
    print(res)


# 3.查询数据
def query_entity():
    # 1.单向量查询
    res = client.search(
        collection_name="demo_v2",
        # 向量数据，由query转成的向量数据
        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
        # 限定条数
        limit=2,
        # 搜索条件
        search_params={"metric_type": "COSINE"},
        # 输出字段
        output_fields=["id", "vector", "color"],
        # 搜索字段
        anns_field="vector"
    )
    print(res)

    # 2.多向量查询：了解，用得不多
    res = client.search(
        collection_name="demo_v2",
        data=[
            [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
            [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
        ],
        limit=2,
        search_params={"metric_type": "COSINE"},
        output_fields=["id", "vector", "color"],
        anns_field="vector"
    )
    print(res)

    # 3.过滤查询
    res = client.search(
        collection_name="demo_v2",
        # 向量数据，由query转成的向量数据
        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
        # 限定条数
        limit=2,
        # 搜索条件
        search_params={"metric_type": "COSINE"},
        # 输出字段
        output_fields=["id", "vector", "color"],
        # 搜索字段
        anns_field="vector",
        filter="color like 'red_9%'"
    )
    print(res)

    # 4.范围搜索
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "radius": 0.1,
            "range_filter": 1
        }
    }
    res = client.search(
        collection_name="demo_v2",
        # 向量数据，由query转成的向量数据
        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
        # 限定条数
        limit=2,
        # 搜索条件
        search_params=search_params,
        # 输出字段
        output_fields=["id", "vector", "color"],
        # 搜索字段
        anns_field="vector"
    )
    print(res)


if __name__ == '__main__':
    # create_collection()
    # insert_entity()
    # upsert_entity()
    query_entity()
