from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="mxbai-embed-large")

# 向量数据库已经存在，那么可以直接加载
chromadaDB = Chroma(persist_directory='./chroma_db', embedding_function=embedding)

# 使用向量数据库进行查询
query = "1917年发生了什么？"
# result = chromadaDB.similarity_search(query, k=2)

# 使用 as_retriever 方法返回 Retriever 对象，然后调用 invoke 方法进行查询
retriever = chromadaDB.as_retriever(search_kwargs={"k":2})
result = retriever.invoke(query)
print(f'result-->{result}')