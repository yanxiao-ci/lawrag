"""
    目的：学习ChatPromptTemplate
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")

# 定义提示词模版
template_str = "帮我讲个关于{name}笑话吧"
# 创建模板
prompt_template = ChatPromptTemplate.from_template(template_str)
# 填充模版，获取完整提示词
prompt = prompt_template.format_messages(name="气球")
print(f'prompt-->{prompt}')

# 调用模型
result = model.invoke(prompt)
print(f'result-->{result}')