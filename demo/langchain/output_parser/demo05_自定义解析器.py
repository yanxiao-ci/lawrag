from langchain_core.output_parsers import BaseOutputParser
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="deepseek-r1:1.5b")

class CustomKeyValueParser(BaseOutputParser[Dict[str, Any]]):
    """解析形如'key: value'的文本"""
    def parse(self, text: str) -> Dict[str, Any]:
        """从文本中解析键值对"""
        result = {}
        lines = text.strip().split('\n')

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()

        return result

    def get_format_instructions(self) -> str:
        """提供格式指导给模型"""
        return """请以'键: 值'的格式返回信息，每行一个键值对。
                例如：
                名称: 爱因斯坦
                职业: 物理学家
                贡献: 相对论
                出生年月: 1959-095
                """


# 使用自定义解析器
custom_parser = CustomKeyValueParser()
custom_prompt = ChatPromptTemplate.from_template(
    "提供关于{person}的基本信息。\n{format_instructions}"
)

# 组合组件
custom_chain = custom_prompt | model | custom_parser

# 调用模型
result = custom_chain.invoke({
    "person": "袁隆平",
    "format_instructions": custom_parser.get_format_instructions()
})
print(result)