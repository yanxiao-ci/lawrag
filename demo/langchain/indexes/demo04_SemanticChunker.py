from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
#语义切割  用的少
# mxbai-embed-large：向量模型
embed = OllamaEmbeddings(model="mxbai-embed-large")

text_splitter = SemanticChunker(
    embeddings=embed,
    # 百分比
    breakpoint_threshold_type='percentile',
    # 更低的百分位会更“积极”地切分（数值越小 => 切得越多）
    breakpoint_threshold_amount=70.0,
    # 句子拆分正则为同时识别中/英文终结符
    sentence_split_regex=r'(?<=[。！？.!?])\s*',
    # 每个切分后片段（chunk）至少要有多少个字符
    min_chunk_size=10
)

text = """
人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
它们可以帮助人们进行写作、代码生成、甚至是科研探索。
相比之下，新能源的发展同样重要。
电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。
"""

docs = text_splitter.split_text(text)
for i, d in enumerate(docs):
    print(f"------ Chunk {i + 1} ------")
    print(d.strip())
    print()
