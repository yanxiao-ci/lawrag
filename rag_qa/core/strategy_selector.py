"""
    策略选择器：rag/core/strategy_selector.py
"""
# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
from base.config import config
from base.logger import logger
# 导入 OpenAI
from openai import OpenAI


class StrategySelector:
    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                             base_url=config.DASHSCOPE_BASE_URL)
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()


    # 定义私有方法，获取策略选择 Prompt 模板
    def _get_strategy_prompt(self):
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：劳动合同解除后的经济补偿怎么算？
                        * 策略：直接检索
                        * 查询：《行政处罚法》第五十七条讲什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：平台经济对劳动者权益保护带来了哪些影响？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：劳动合同法和劳动法在加班费规定上有什么不同？
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我在网上买到了假货，商家拒绝退款，平台也不处理，这种情况我该怎么维权？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    # 调用大模型
    def call_dashscope(self, prompt):
        try:
            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0 #值越小,每次回答越集中
            )
            # 返回完成结果
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"


    # 定义方法，选择检索策略
    def select_strategy(self, query):
        #   调用 LLM 获取检索策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        # ['直接检索', '假设问题检索', '子查询检索', '回溯问题检索']
        return strategy


if __name__ == '__main__':
    ss = StrategySelector()
    ss.select_strategy('劳动合同解除后的经济补偿怎么算？')
    # ss.select_strategy('平台经济对劳动者权益保护带来了哪些影响？')
    # ss.select_strategy('劳动合同法和劳动法在加班费规定上有什么不同？')
    # ss.select_strategy('我在网上买到了假货，商家拒绝退款，平台也不处理，这种情况我该怎么维权？')
