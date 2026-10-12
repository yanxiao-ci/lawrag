"""
    目的：StrOutputParser的使用
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")
# 创建提示词模板
prompt = ChatPromptTemplate.from_template("解释{topic}是什么？回答控制20字以内")

chain = prompt | model
# 调用大模型
result = chain.invoke({"topic": "ai"})
print(f'原生的result:{result}')


# 创建 StrOutputParser
parser = StrOutputParser()
# 组装链
chain = prompt | model | parser
# 调用大模型
result = chain.invoke({"topic": "ai"})
# 输出结果
print(f'StrOutputParser result:{result}')
