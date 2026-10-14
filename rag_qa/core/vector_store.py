"""
    文档向量化与存储：core/vector_store.py
"""
from base.config import config
from base.logger import logger
import os
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
# 导入 torch，用于自动选择推理设备（有 GPU 用 GPU，否则回退 CPU）
import torch


# 第三召回源：law_faq 的 20,000 条法律问答单独放一个集合，不混进法条集合
# （法条集合的检索强制 status == "有效"，FAQ 条目没有 status，混进去会被过滤掉）
FAQ_COLLECTION_NAME = "lawfaq"


# # 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=config.MILVUS_COLLECTION_NAME,
                 host=config.MILVUS_HOST,
                 port=config.MILVUS_PORT,
                 database=config.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 重排序 模型路径
        rerank_model_path = os.path.join(config.MODELS_DIR, 'bge-reranker-large')
        # device 代表推理设备：cuda（NVIDIA 显卡）/ mps（苹果 M 系）/ cpu。自动探测，无需手改。
        # 有 GPU 时向量化提速一两个数量级（21k 条法条：CPU 约 3 小时 -> GPU 几分钟）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # GPU 上开 FP16：显存占用约减半（bge-m3 + reranker 合计约 2.2GB），8GB 显存足够；
        # CPU 上保持 FP32 更稳（fp16 在 CPU 上容易触发 LayerNormKernelImpl 报错）。
        use_f16 = self.device == "cuda"
        logger.info(f"向量化/重排 使用设备: {self.device} (fp16={use_f16})")
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        self.reranker = CrossEncoder(rerank_model_path, device=self.device)
        # bge-m3 模型路径
        bge_m3_model_path = os.path.join(config.MODELS_DIR, 'bge-m3')
        # 初始化 BGE-M3 嵌入函数
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_model_path,
            use_fp16=use_f16,
            device=self.device,
            # 批量大小：默认 16 在 CPU/GPU 上都偏小，批到 64 吞吐更高（该库内部按 batch_size 分批前向）
            batch_size=64,
        )
        # 获取稠密向量的维度 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f'http://{self.host}:{self.port}', db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()
        # 确保「法律问答」集合存在（空集合也先建好，避免在线召回时报错）
        self._ensure_faq_collection()

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加来源字段 source（当前只有 law），VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            # P1：法条元数据字段（检索时用 status 过滤已废止/旧版本，chapter 用于层级展示）
            schema.add_field(field_name="law_name", datatype=DataType.VARCHAR, max_length=120)
            schema.add_field(field_name="article_no", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=200)
            schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=12)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                # 丢弃权值最小的 20% 的维度（只保留权值较大的 80%）
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义私有方法：创建或加载「法律问答」集合（第三召回源）
    def _ensure_faq_collection(self):
        if self.client.has_collection(FAQ_COLLECTION_NAME):
            self.client.load_collection(FAQ_COLLECTION_NAME)
            logger.info(f"已加载问答集合 {FAQ_COLLECTION_NAME}")
            return
        # 问答集合的 Schema 比法条简单：没有条/章/时效这些字段
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="domain", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name="faq_dense_index",
                               index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
        index_params.add_index(field_name="sparse_vector", index_name="faq_sparse_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                               params={"drop_ratio_build": 0.2})
        self.client.create_collection(collection_name=FAQ_COLLECTION_NAME, schema=schema,
                                      index_params=index_params)
        self.client.load_collection(FAQ_COLLECTION_NAME)
        logger.info(f"已创建空问答集合 {FAQ_COLLECTION_NAME}（用 tools/build_lawfaq_index.py 灌数据）")

    # 定义方法：把 law_faq 的问答写进问答集合（供建索引脚本调用，分批向量化）
    def add_faq_documents(self, records, batch_size=256):
        """records: [{'question':..,'answer':..,'domain':..}, ...]"""
        total = 0
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            texts = [f"{r['question']}\n{r['answer']}"[:20000] for r in batch]
            embeddings = self.embedding_function(texts)
            data = []
            for i, (rec, text) in enumerate(zip(batch, texts)):
                row = embeddings["sparse"][i]
                indices = row.col if hasattr(row, "col") else row.indices
                data.append({
                    "id": hashlib.md5(text.encode("utf-8")).hexdigest(),
                    "text": text,
                    # fp16 推理 -> float32，和集合里的向量类型保持一致
                    "dense_vector": embeddings["dense"][i].astype("float32"),
                    "sparse_vector": {int(j): float(v) for j, v in zip(indices, row.data)},
                    "question": rec["question"][:2000],
                    "answer": rec["answer"][:20000],
                    "domain": (rec.get("domain") or "")[:50],
                    "source": "faq",
                })
            self.client.upsert(collection_name=FAQ_COLLECTION_NAME, data=data)
            total += len(data)
            logger.info(f"问答集合已写入 {total}/{len(records)} 条")
        self.client.flush(collection_name=FAQ_COLLECTION_NAME)
        return total

    # 定义方法：在问答集合里召回（第三召回源，稠密 + 稀疏混合检索）
    def search_faq(self, query, k=3):
        if not self.client.has_collection(FAQ_COLLECTION_NAME):
            logger.warning(f"问答集合不存在: {FAQ_COLLECTION_NAME}，跳过 FAQ 召回")
            return []
        embeddings = self.embedding_function([query])
        dense_vector = embeddings["dense"][0].astype("float32")
        row = embeddings["sparse"][0]
        indices = row.col if hasattr(row, "col") else row.indices
        sparse_vector = {int(i): float(v) for i, v in zip(indices, row.data)}
        reqs = [
            AnnSearchRequest(data=[dense_vector], anns_field="dense_vector",
                             param={"metric_type": "IP", "params": {"nprobe": 10}}, limit=k),
            AnnSearchRequest(data=[sparse_vector], anns_field="sparse_vector",
                             param={"metric_type": "IP", "params": {}}, limit=k),
        ]
        hits = self.client.hybrid_search(
            collection_name=FAQ_COLLECTION_NAME, reqs=reqs,
            ranker=WeightedRanker(1.0, 0.7), limit=k,
            output_fields=["text", "question", "answer", "domain", "source"],
        )[0]
        docs = []
        for hit in hits:
            e = hit["entity"]
            docs.append(Document(page_content=e["text"], metadata={
                "source": e.get("source", "faq"),
                "question": e.get("question", ""),
                "answer": e.get("answer", ""),
                "domain": e.get("domain", ""),
                "score": hit.get("distance"),
            }))
        return docs

    # 定义方法，存储文档到向量数据库
    def add_documents(self, documents):
        # collection中一条数据
        data = []
        # 提取文档内容
        texts = [doc.page_content for doc in documents]
        # BGM-M3 生成嵌入【向量化】
        embeddings = self.embedding_function(texts)
        # 遍历文档
        for i, doc in enumerate(documents):
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            # 初始化稀疏向量字典
            sparse_vector = {}
            try:
                # 新版本 milvus-model 使用 coo_array 格式
                row = embeddings["sparse"][i]
                # 获取非零元素的列索引数组
                if hasattr(row, 'col'):  # coo_array 格式，新版Milvus
                    indices = row.col
                else:  # csr_matrix 格式
                    indices = row.indices
            except Exception as e:
                # 兼容旧版本 milvus-model
                row = embeddings["sparse"].getrow(0)
                indices = row.indices
            # 获取稀疏向量的非零值
            values = row.data
            # logger.info(f"稀疏向量的非零索引：{indices}:{values}")
            # logger.info("=" * 100)
            for idx, value in zip(indices, values):
                # fp16 推理时 value 是 float16，Milvus 只收 float32，这里统一转换
                sparse_vector[int(idx)] = float(value)
            # 构建数据记录
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                # 同上：fp16 推理出来的稠密向量要转 float32 才能入 Milvus
                "dense_vector": embeddings["dense"][i].astype("float32"),
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                # 无法获取到source数据，则当空串处理
                "source": doc.metadata.get("source", ""),
                "timestamp": doc.metadata.get("timestamp", ""),
                # P1：法条元数据（非法律语料为空串）
                "law_name": doc.metadata.get("law_name", ""),
                "article_no": doc.metadata.get("article_no", ""),
                "chapter": doc.metadata.get("chapter", ""),
                "status": doc.metadata.get("status", ""),
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f"存储 {len(documents)} 个文档到向量数据库...")
        else:
            logger.error("没有加载到数据")

    # 定义方法：混合检索（粗排），返回 top-k 子块 Document；评测 Recall@k 直接复用
    def search_children(self, query, k=config.RETRIEVAL_K, source_filter=None,
                        effective_only=True, use_sparse=True):
        """混合检索（粗排），返回 top-k 子块 Document。

        effective_only / use_sparse 是给评测做消融用的开关，默认值等于线上行为：
          effective_only=False  -> 对照组：不加 status == "有效" 过滤（看时效过滤的代价）
          use_sparse=False      -> 对照组：只走向量（稠密），看 BGE-M3 稀疏分支的贡献
        """
        # BGM-M3 生成嵌入【向量化】
        query_embeddings = self.embedding_function([query])
        # 稠密向量（fp16 推理 -> 转 float32，与集合里存的向量类型保持一致）
        dense_query_vector = query_embeddings["dense"][0].astype("float32")
        # 稀疏向量
        sparse_vector = {}
        row = query_embeddings["sparse"][0]
        indices = row.col if hasattr(row, 'col') else row.indices
        for idx, value in zip(indices, row.data):
            sparse_vector[int(idx)] = float(value)
        # 构建查询条件：来源（source）过滤 + P1 时效过滤
        # 法律语料另加 status == "有效"：知识库里同时留了历史版本用于对照，
        # 但检索（在线问答）只应该命中现行有效版本，避免答出已修订/已废止条文。
        filter_parts = []
        if source_filter:
            filter_parts.append(f'source == "{source_filter}"')
        # P1 收尾：时效过滤不再挂在 source_filter 下面。
        # 前端「来源」默认是「全部」，此时 source_filter 为空，原来会连时效过滤一起跳过，
        # 实测“专利侵权赔偿数额如何确定”的粗排前 8 条会混进 3 条已修改版本。
        # 本集合目前只有法律语料（21565 条 source 全为 law），所以直接常开。
        if effective_only:
            filter_parts.append('status == "有效"')
        fileter_expr = " and ".join(filter_parts)
        # 构建查询条件
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=fileter_expr
        )
        reqs = [dense_request]
        if use_sparse:
            sparse_request = AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=k,
                expr=fileter_expr
            )
            reqs.append(sparse_request)
        # 指定 稠密和稀疏向量的权重比（只走向量时权重退化成 1.0）
        ranker = WeightedRanker(1.0, 0.7) if use_sparse else WeightedRanker(1.0)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,  # 粗排
            limit=k,
            output_fields=["id", "text", "parent_id", "parent_content", "source", "timestamp",
                           "law_name", "article_no", "chapter", "status"],
        )[0]
        # 判断结果是否为空，如果为空，则直接返回 []
        if not results:
            return []
        # 将结果转换为Document列表
        return [self._doc_from_hit(hit['entity']) for hit in results]

    #定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, source_filter=None):
        logger.info(f"hybrid_search_with_rerank 执行混合检索并重排序...query:{query}, k:{k}, source_filter:{source_filter}")
        # 混合检索（粗排）
        child_chunks = self.search_children(query, k=k, source_filter=source_filter)
        if not child_chunks:
            return []
        logger.info(f"去重前子(父)块数量:{len(child_chunks)}")
        # 对父文档去重
        parent_docs = self._get_unique_parent_docs(child_chunks)
        logger.info(f"去重后父块数量:{len(parent_docs)}")
        # 判断父文档数量<2，返回
        if len(parent_docs) < 2:
            return parent_docs[:config.CANDIDATE_M]
        # >=2，计算分数，倒序排序，返回前M个结果
        pairs = [[query, doc.page_content] for doc in parent_docs]
        # 将问题与答案进行算分
        scores = self.reranker.predict(pairs)  #[0.8, 0.6, 0.9]
        # 根据分数降序排列：[0.9分值的文档， 0.8分值的文档， 0.6分值的文档]
        rank_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # logger.info(f"排序后文档:{rank_parent_docs}")
        logger.info(f"排序后文档数量:{len(rank_parent_docs)}")
        # 返回前 CANDIDATE_M=2 个文档
        return rank_parent_docs[:config.CANDIDATE_M]




    #定义私有方法，从 Milvus 查询结果转换成创建 Document 对象
    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit["text"],   # 子块内容
            metadata={
                "id": hit["id"],
                "parent_id": hit["parent_id"],
                "parent_content": hit["parent_content"],   # 父块内容，将来用于拼接提示词
                "source": hit["source"],
                "timestamp": hit["timestamp"],
                # P1：法条元数据（老集合没有这些字段时取空串）
                "law_name": hit.get("law_name", ""),
                "article_no": hit.get("article_no", ""),
                "chapter": hit.get("chapter", ""),
                "status": hit.get("status", ""),
            }
        )

    # 定义私有方法，从子块中提取去重的父文档，里面只有内容
    def _get_unique_parent_docs(self, child_chunks):
        parent_contents = set()
        for chunk in child_chunks:
            parent_contents.add(chunk.metadata.get('parent_content', chunk.page_content))
        return [Document(page_content=content) for content in parent_contents]

# 测试
if __name__ == '__main__':
    vector_store = VectorStore()
    # from document_processor import process_documents
    # docs = process_documents(config.DATA_DIR + "/law_data")
    # vector_store.add_documents(docs)

    vector_store.hybrid_search_with_rerank(query="租房押金不退怎么办？", k=5, source_filter='law')
