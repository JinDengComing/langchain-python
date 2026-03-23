import os
import sys
import langchainhub as hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils.tools import tools
from llmModel import llm

# 添加项目的site-packages目录到Python路径
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Lib", "site-packages")
    )
)

# 从 LangChain Hub 获取预设 Prompt
prompt = hub.pull("hwchase17/openai-tools-agent")

# 构造并运行 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试运行
response = agent_executor.invoke({"input": "北京天气怎么样？计算 25 乘以 4 等于多少？"})
print(response["output"])
