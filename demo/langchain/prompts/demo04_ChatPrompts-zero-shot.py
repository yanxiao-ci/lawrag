"""
    目的：学习ChatPromptTemplate zero-shot
"""
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")

# 系统信息
system_prompt = SystemMessage("你是取名专家。")
# 用户信息模版
human_str = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字。"
human_template = HumanMessagePromptTemplate.from_template(human_str)
# 组装模板
chat_template = ChatPromptTemplate.from_messages([system_prompt, human_template])
# 生成最终的提示词
prompt = chat_template.format_messages(lastname="郭")
print(f'prompt-->{prompt}')

# 调用模型
result = model.invoke(prompt)  # 返回结果出了content外，还有元数据信息
print(f'result-->{result}')