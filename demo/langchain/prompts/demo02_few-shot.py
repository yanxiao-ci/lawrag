"""
    目的：学习PromptTemplate few-shot
"""
# from langchain import PromptTemplate, FewShotPromptTemplate # Langchain 0.x版本使用
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="deepseek-r1:1.5b")

# 示例样式列表
examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"}
]

# 定义提示词模板字符串
example_template = """
单词: {word}
反义词: {antonym}\\n
"""
# 基于模板填充，获取完整提示词
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template,
)

# 创建FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个单词的反义词",
    suffix="单词: {input}\\n反义词:",
    input_variables=["input"],
    example_separator="\\n",
)

prompt_text = few_shot_prompt.format(input="瞬间")
print(prompt_text)
print("=" * 50)
result = model.invoke(prompt_text)
print(result)
# 细