from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="qwen3.5:4b")

# 定义Pydantic模型  BaseModel是基础模型,必须继承
class Movie(BaseModel):
    # 属性名: 类型 = 字段(描述)
    title: str = Field(description="电影标题")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")
    genre: List[str] = Field(description="电影类型")
    rating: float = Field(description="评分（1-10）")


# 创建Pydantic解析器
pydantic_parser = PydanticOutputParser(pydantic_object=Movie)

# 创建带格式说明的提示模板
format_instructions = pydantic_parser.get_format_instructions()
pydantic_prompt = ChatPromptTemplate.from_template(
    "生成环太平洋的{genre}电影的信息。\n{format_instructions}"
)

# 组合组件
pydantic_chain = pydantic_prompt | model | pydantic_parser

# 调用链
movie_data = pydantic_chain.invoke({
    "genre": "科幻",
    "format_instructions": format_instructions
})
print(movie_data)