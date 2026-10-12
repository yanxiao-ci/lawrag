"""
    目的：学习PromptTemplate zero-shot
    需求：我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字
"""
# 1. 导入包
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 2. 创建模型
model = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://127.0.0.1:11434")
# 3. 提示词字符串
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
# 4. 创建提示词模型
prompt = PromptTemplate(
    template=template,
    input_variables=["lastname"],
)
# 5. 获取完整提示词
prompt_text = prompt.format(lastname="张")
print(prompt_text)
# 6. 调用大模型
result = model.invoke(prompt_text)
print(result)