"""
    文档处理器：rag_qa/core/document_processor.py
    负责加载和分割文档
"""
import os
from base.config import config
from base.logger import logger
from datetime import datetime
from rag_qa.edu_document_loaders.edu_docloader import OCRDOCLoader
from rag_qa.edu_document_loaders.edu_imgloader import OCRIMGLoader
from rag_qa.edu_document_loaders.edu_pdfloader import OCRPDFLoader
from rag_qa.edu_document_loaders.edu_pptloader import OCRPPTLoader

from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


# 定义函数，从指定文件夹加载多种类型的文件，并添加元数据
def load_documents_from_directory(directory_path):
    # 初始化空列表，用于存储加载后的文档
    documents = []
    #os.work()递归遍历目录树
    # 获取支持的文件扩展名
    supported_extensions = document_loaders.keys()
    # 获取directory_path文件名_前面的内容
    source = os.path.basename(directory_path).replace("_data", "")
    logger.info(f"source:{source}")
    # 获取所有目录中的文件
    '''
        参数：
            root:  当前正在遍历的目录路径（字符串）
            _   :  当前目录下的子目录列表（列表）
            files:  当前目录下的文件列表（列表）
        os.walk:  这是一个生成器函数，会递归遍历目录树
    '''
    for root, _, files in os.walk(directory_path):
        for file in files:
            # 获取file的绝对路径
            file_path = os.path.join(root, file)
            # 获取file的扩展名
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in supported_extensions:
                load_class = document_loaders[file_extension]
                # txt文件需要指定字符集
                if file_extension == '.txt':
                    loader = load_class(file_path, encoding='utf-8')
                else:
                    loader = load_class(file_path)
                # 加载文档，一个文档加载完成后会封装成一个list
                load_docs = loader.load()
                # print(f"load_docs:{type(load_docs)}")
                for doc in load_docs:
                    # 添加元数据
                    doc.metadata["source"] = source
                    doc.metadata["file_path"] = file_path
                    doc.metadata["timestamp"] = datetime.now().isoformat()
                    # documents.append(doc)
                documents.extend(load_docs)
                logger.info(f'加载文件成功:{file_path}')
            else:
                logger.error(f"不支持的文件类型:{file_extension}")
                continue

    return documents


# 处理文档并进行分层拆分，返回子块结果
def process_documents(directory_path,
                      parent_chunk_size=config.PARENT_CHUNK_SIZE,
                      child_chunk_size=config.CHILD_CHUNK_SIZE,
                      parent_chunk_overlap=config.PARENT_CHUNK_OVERLAP,
                      child_chunk_overlap=config.CHILD_CHUNK_OVERLAP):
    # 子块文档列表
    child_chunks = []
    # 加载指定目录下所有的文档
    documents = load_documents_from_directory(directory_path)
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)

    for i, doc in enumerate(documents):
        # 获取文档的扩展名
        file_extension = os.path.splitext(doc.metadata["file_path"])[1].lower()
        # 选择分割器
        is_markdown = file_extension == ".md"
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        # 获取所有的父块文档
        parent_docs = parent_splitter_to_use.split_documents([doc])
        for j, parent_doc in enumerate(parent_docs):
            # 为父块文档添加元数据
            # 父块ID doc_{i}_parent_{j}
            parent_doc.metadata["id"] = f'doc_{i}_parent_{j}'
            parent_doc.metadata["parent_content"] = parent_doc.page_content
            # 获取所有的子块文档
            child_docs = child_splitter_to_use.split_documents([parent_doc])
            # 遍历子块文档
            for k, child_doc in enumerate(child_docs):
                # 为子块文档添加元数据
                child_doc.metadata["parent_id"] = parent_doc.metadata["id"]
                child_doc.metadata["parent_content"] = parent_doc.page_content
                # 子块ID doc_{i}_parent_{j}_child_{k}
                child_doc.metadata["id"] = parent_doc.metadata["id"] + f'_child_{k}'
                child_chunks.append(child_doc)
                logger.info(f"处理文档成功:父块{parent_doc.metadata['id']}:子块:{child_doc.metadata['id']}")
                logger.warning('=' * 100)
        print("\n\n")
    # 返回所有子块列表
    return child_chunks


if __name__ == "__main__":
    # chunks = load_documents_from_directory(f"{config.DATA_DIR}/ai_data")
    chunks = process_documents(f"{config.DATA_DIR}/ai_data")
    print(chunks)