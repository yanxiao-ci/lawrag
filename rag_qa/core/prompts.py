"""
    提示词模板：rag_qa/core/prompts.py
"""
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from base.config import config


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:

    # 定义 RAG 提示模板
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
                你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：
        
                1. **分析问题和上下文**：
                   - 基于提供的上下文（如果有）和你的知识回答问题。
                   - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。
        
                2. **评估对话历史**：
                   - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
                   - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
                   - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。
        
                3. **生成回答**：
                   - 提供清晰、准确的回答，避免无关信息。
                   - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
                   - 如果前面判断是通用问题，请大模型自己回答。
        
                **上下文**: {context}
                **对话历史**:
                {history}
                **问题**: {question}
        
                **回答**:
                """,
            input_variables=["context", "history", "question", "phone"],
        )

    # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        # LLM生成一个假设答案，后期在进行检索时，会将假设答案作为用户的query到RAG中检索
        return PromptTemplate(
            template="""  
                假设你是用户，想了解以下问题，请生成一个简短的假设答案：
                问题: {query}
                假设答案:
                """,
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        # 将用户的原始问题，进行拆分，拆分成多个子问题
        return PromptTemplate(
            template="""  
                将以下复杂查询分解为多个简单子查询，每行一个子查询：
                查询: {query}
                子查询:
                """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        # 将用户的原始问题简化为一个简单问题，后期通过简单问题去RAG中检索
        return PromptTemplate(
            template="""  
                将以下复杂查询简化为一个更简单的问题：
                查询: {query}
                简化问题:
                """,
            #   定义输入变量
            input_variables=["query"],
        )


if __name__ == '__main__':
    # query = "人工智能在教育领域的应用有哪些？"
    # prompt = RAGPrompts.hyde_prompt().format(query=query)

    query = "AI大模型学科和AI应用与智能体学科学费分别是多少？"
    prompt = RAGPrompts.subquery_prompt().format(query=query)
    print(prompt)

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                    base_url=config.DASHSCOPE_BASE_URL)
    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )
    # 返回完成结果
    print(completion.choices[0].message.content)
