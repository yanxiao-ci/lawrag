"""
    目的：学习CommaSeparatedListOutputParser
"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen2.5:7b")
# 创建列表解析器
parser = CommaSeparatedListOutputParser()
# 创建带格式说明的提示模板
format_instructions = parser.get_format_instructions()
prompt = ChatPromptTemplate.from_template(
    "用中文列出{topic}的五个最重要特点。\n{format_instructions}"
)

# 组合组件
chain = prompt | model | parser

# 调用链
result = chain.invoke({
    "topic": "人类",
    "format_instructions": format_instructions
})
print(result)