"""
    加载非结构化文档，以txt为例
"""
from langchain_unstructured import UnstructuredLoader

# 创建 UnstructuredLoader 对象
loader = UnstructuredLoader('./data/衣服属性.txt', encoding='utf-8')
# 加载文档
docs = loader.load()
print(f'docs-->{docs}')
print(f'len-->{len(docs)}')
print(f'第一行数据-->{docs[0].page_content}')#非结构化,一行即是一个文档
print('*' * 100)

from langchain_community.document_loaders import TextLoader

# 创建 TextLoader 对象
loader = TextLoader('./data/衣服属性.txt', encoding='utf-8')
docs = loader.load()
print(f'docs-->{docs}')
print(f'len-->{len(docs)}')
print('第一行数据-->{}'.format(docs[0].page_content.split('\n')[0]))#全部内容一个文档