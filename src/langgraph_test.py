import operator
import os
import sys

# 添加项目的site-packages目录到Python路径
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Lib", "site-packages")
    )
)

from dotenv import load_dotenv
from typing import Annotated, List, TypedDict, Union
import asyncio
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()


# 1. 定义状态 (State)
class AgentState(TypedDict):
    # 使用 Annotated[..., operator.add] 允许消息列表不断累加
    messages: Annotated[List[BaseMessage], operator.add]


# 2. 初始化模型
llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0.7)

# 3. 定义节点逻辑


def generation_node(state: AgentState):
    """生成回答，并提取 DeepSeek 的思考过程"""

    # 调用模型
    response = llm.invoke(state["messages"])

    # 提取思考过程 (Reasoning Content)
    # 不同的集成版本可能路径略有不同，通常在 additional_kwargs 中
    reasoning = response.additional_kwargs.get("reasoning_content", "")

    print("\n" + "=" * 20 + " 模型思考中 " + "=" * 20)
    if reasoning:
        print(reasoning)
    else:
        print("（本次输出未包含显式推理链）")
    print("=" * 50 + "\n")

    return {"messages": [response]}


def reflection_node(state: AgentState):
    """作为批评者，对上一个回答提出改进建议"""
    last_message = state["messages"][-1]
    reflection_prompt = f"请评价以下内容并指出改进点（如果已完美请回复'DONE'）：\n\n{last_message.content}"

    # 构造反思消息
    reflection = llm.invoke([HumanMessage(content=reflection_prompt)])
    return {"messages": [reflection]}


def should_continue(state: AgentState):
    """决策器：判断是结束还是继续循环"""
    last_message = state["messages"][-1]
    if "DONE" in last_message.content.upper() or len(state["messages"]) > 6:
        return END
    return "generate"


# 4. 构建图 (Graph)
workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

# 添加节点
workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)

# 设置入口
workflow.set_entry_point("generate")

# 添加连线
workflow.add_edge("generate", "reflect")

# 添加条件循环
workflow.add_conditional_edges(
    "reflect", should_continue, {"generate": "generate", END: END}
)

# 5. 编译与运行
app = workflow.compile()

# 测试：写一首关于量子物理的诗
inputs = {"messages": [HumanMessage(content="请比较python和javascript，要求通俗易懂。")]}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         print(f"--- Node: {key} ---")
#         print(value["messages"][-1].content)
#         print("-" * 20)


async def run_agent_with_reasoning():

    # 使用 astream_events 获取内部 token 流
    async for event in app.astream_events(inputs, version="v2"):
        kind = event["event"]

        # 监听模型输出的 chunk
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            # 检查 chunk 中是否带有推理内容 (部分版本支持直接流式获取 reasoning)
            reasoning_chunk = event["data"]["chunk"].additional_kwargs.get(
                "reasoning_content", ""
            )

            if reasoning_chunk:
                print(reasoning_chunk, end="", flush=True)  # 实时打印思考
            if content:
                # 思考结束后打印正式回答
                print(content, end="", flush=True)
        # 2. 捕获节点切换（可选，方便调试）
        elif kind == "on_chain_start" and event["name"] == "reflect":
            print("\n\n--- ⚖️ 进入反思阶段 ---")


# 运行异步函数
if __name__ == "__main__":
    try:
        # 尝试获取当前正在运行的循环
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果在 Notebook 等环境，直接创建任务
            loop.create_task(run_agent_with_reasoning())
        else:
            loop.run_until_complete(run_agent_with_reasoning())
    except RuntimeError:
        # 如果彻底没有循环，则新建
        asyncio.run(run_agent_with_reasoning())
