"""
    目的：学习ChatPromptTemplate few-shot
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

# 实例化模型
model = OllamaLLM(model="deepseek-r1:1.5b")

# 创建 prompt 模版
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个语文老师，给出每个单词的反义词"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)
# 创建 few-shot prompt
# history = [
#     HumanMessage(content="开心"),
#     AIMessage(content="难过"),
#     HumanMessage(content="高"),
#     AIMessage(content="矮")
# ]
history = [("human", "开心"), ("ai", "难过"), ("human", "高"), ("ai", "矮")]
prompt = prompt_template.format_messages(history=history, question="白富美")
print(f"prompt-->{prompt}")

# 调用模型
result = model.invoke(prompt)
print(f'result-->{result}')