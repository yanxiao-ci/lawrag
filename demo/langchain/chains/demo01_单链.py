"""
    目的：单链组合
"""
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
llm = OllamaLLM(model="qwen2.5:7b")

# 定义模板字符串
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
# 创建模板
prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template)

# 组合链,上一步的输出是下一步的输入
chain = prompt | llm

# 执行链
print(chain.invoke("王"))
# print(chain.invoke(input="王"))
