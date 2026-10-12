from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools

# 实例化模型，该模型必须支持 function call
model = ChatOllama(model="qwen3.5:4b")

# 基于内存的记忆
memory = MemorySaver()
# 加载工具
tools = load_tools(['wikipedia', 'llm-math'], llm=model)
# 创建智能体
agent_executor = create_agent(model, tools, checkpointer=memory)

# 配置，线程ID    记忆必须结合 config的 thread_id 才能实现
config = {"configurable": {"thread_id": "abc123"}} #会话唯一标记

# 流式输出
for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="计算一下300的25%是多少？")]},
        config
):
    print(chunk)
    print("=" * 50)
    print()
