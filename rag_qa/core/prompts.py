"""
    提示词模板：rag_qa/core/prompts.py
"""
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from base.config import config


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:

    # 定义 RAG 提示模板（法律助手 + 强制引用格式）
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
                你是一名严谨的法律法规智能问答助手，依据给定的“检索上下文”回答用户问题。

                上下文里可能有两种块，都算依据：
                  ·【命中条文】= 法条原文，主要依据；
                  ·【命中问答】= 知识库里相近问题的问答对，可作参考依据
                    （其中引用的法条同样算依据，引用时仍要按《法律名》第X条 的格式写清楚）。

                【回答要求】
                1. 必须严格依据“检索上下文”作答，不得编造法律名称或条文；
                   若上下文没有相关依据，请回复：“未能在检索到的条文中找到依据，请补充法律名称或条文号；如需人工协助，电话：{phone}。”
                2. 每一条法律结论都要标注来源，引用格式统一为《法律名》第X条，
                   例如：根据《中华人民共和国民法典》第一千零六十二条，……；
                   上下文里若标了编/章/节，可一并写出，例如（第五编 婚姻家庭 第三章 家庭关系）；
                   只引用标注为「有效」的条文，不要引用标注为「已修改 / 已废止」的旧版本。
                3. 只引用上下文中真实出现的法律与条文，禁止臆造条号；无法确定条号时不要引用。
                4. 若上下文不足以回答，可以先说明“法条未覆盖”，再结合常识做一般性说明，并明确区分“法律规定”与“一般说明”。
                5. 语言简洁、条理清晰，先给结论再给依据。

                **检索上下文**:
                {context}

                **对话历史**:
                {history}

                **用户问题**: {question}

                **回答**:
                """,
            input_variables=["context", "history", "question", "phone"],
        )

    # 通用知识问答模板（不要求引用法条，避免法律兜底话术误伤通用问题）
    @staticmethod
    def general_prompt():
        return PromptTemplate(
            template="""
                你是一个乐于助人的中文智能助手，负责回答与法律法规无关的通用问题。

                【回答要求】
                1. 基于你自己的知识准确、简洁地回答，条理清晰。
                2. 不要编造法律条文，也不要引用《法律名》第 X 条。
                3. 如果问题其实是一个具体的法律问题（例如合同、纠纷、维权、工伤等），
                   就说明这属于法律问题，建议用户补充涉及的法律名称或改为咨询法律问题，不要强行作答。
                4. 不要输出与问题无关的客套话。

                **对话历史**:
                {history}

                **用户问题**: {question}

                **回答**:
                """,
            input_variables=["history", "question"],
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
    # query = "平台经济对劳动者权益保护带来了哪些影响？"
    # prompt = RAGPrompts.hyde_prompt().format(query=query)

    query = "劳动合同法和劳动法在加班费规定上有什么不同？"
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
