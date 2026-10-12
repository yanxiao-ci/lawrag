"""
    RecursiveCharacterTextSplitter
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 递归分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,  # 每个块最多 20 个字符
    chunk_overlap=6,  # 相邻分块会共享 6 个字符
    length_function=len,  # 用字符数来衡量长度
    separators=["\n\n", "\n", " ", ""]  # 会优先尝试按 \n\n 分段，如果太长，再按 \n → 空格 → 逐字符切分。
)

text = """
人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
它们可以帮助人们进行写作、代码生成、甚至是科研探索。
相比之下，新能源的发展同样重要。
电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。
"""
docs = text_splitter.split_text(text)
print(docs)
