"""
    目的：多链组合
"""
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
llm = OllamaLLM(model="qwen2.5:7b")
# 创建第一条链
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
first_prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template, )
first_chain = first_prompt | llm

# =========================

# 创建第二条链
second_prompt = PromptTemplate(
    input_variables=["child_name"],
    template="邻居的儿子名字叫{child_name}，给他起一个小名")

second_chain = second_prompt | llm

# 链接两条链
overall_chain = first_chain | second_chain

print(overall_chain)
print('=' * 80)
# 执行链，只需要传入第一个参数
catchphrase = overall_chain.invoke(input="赵")
print(catchphrase)