# """
#     Chroma作为向量数据库实现 向量化存储和检索
# """
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
#
# 1.加载文档
# 创建 TextLoader 对象
loader = TextLoader('./data/pku.txt', encoding='utf-8')
# 加载文档
docs = loader.load()
print(f'docs-->{docs}')

# 2.将文档进行分块
# 创建 CharacterTextSplitter 对象
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=30)
# 分块
split_docs = text_splitter.split_documents(docs)
print(f'split_docs-->{split_docs}')

# # 3.将分割后的文档存储到向量数据库中
# # 加载embedding模型
embedding = OllamaEmbeddings(model="mxbai-embed-large")
# 创建向量数据库，需要指定 存储的文档和向量模型名称以及持久化目录
chromadaDB = Chroma.from_documents(documents=split_docs,
                                   embedding=embedding,
                                   persist_directory='./chroma_db')

# 假如你的向量数据库已经存在，那么可以直接加载
chromadaDB = Chroma(persist_directory='./chroma_db', embedding_function=embedding)

# 4.使用向量数据库进行查询
query = "1937年北京大学发生了什么？"
result = chromadaDB.similarity_search(query, k=1)
print(f'result-->{result}')