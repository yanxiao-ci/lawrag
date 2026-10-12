"""
    目的：学习JsonOutputParser
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen3.5:4b")

# 创建JSON解析器
json_parser = JsonOutputParser()

# 创建带格式说明的提示模板
json_format_instructions = json_parser.get_format_instructions()
json_prompt = ChatPromptTemplate.from_template(
    "生成一个包含{person}基本信息的JSON。应包括姓名、职业、年龄和技能列表, 不要包含任何注释或额外说明。\n{format_instructions}"
)

# 组合组件
json_chain = json_prompt | model | json_parser

# 调用链
result = json_chain.invoke({
    "person": "刘健",
    "format_instructions": json_format_instructions
})

print(result)