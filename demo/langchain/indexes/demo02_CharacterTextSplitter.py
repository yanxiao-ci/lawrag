"""
    CharacterTextSplitter
"""
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# 创建分词器 separator参数指的是分割的分隔符，
# chunk_size 每个块的大小，
# chunk_overlap 重叠的大小
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=1)

# 一句话分割
result1 = text_splitter.split_text("a b c d e f")
print(f'result1--->{result1}')

# 多句话分割
result2 = text_splitter.create_documents(["a b c d e f", "e f g h"])
print(f'result2--->{result2}')

# 多句话分割
result3 = text_splitter.split_documents([Document(page_content="a b c d e f", metadata={"id": "1"})])
print(f'result3--->{result3}')