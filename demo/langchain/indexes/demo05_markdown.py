"""
    MarkdownHeaderTextSplitter
"""
#按层级切
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
("#", "Header 1"),
("##", "Header 2"),
("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_text = "# Header 1\nSome text\n## Header 2\nMore text\n### Header 3\nEven more text"
docs = markdown_splitter.split_text(markdown_text)
print(docs)

"""
[Document(metadata={'Header 1': 'Header 1'}, page_content='Some text'), Document(metadata={'Header 1': 'Header 1', 'Header 2': 'Header 2'}, page_content='More text'), Document(metadata={'Header 1': 'Header 1', 'Header 2': 'Header 2', 'Header 3': 'Header 3'}, page_content='Even more text')]
"""