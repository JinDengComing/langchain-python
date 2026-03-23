import sys
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
# 加载环境变量
load_dotenv()


# 添加项目的site-packages目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Lib', 'site-packages')))

# 检查是否设置了DeepSeek API密钥
if not os.environ.get('DEEPSEEK_API_KEY'):
    print("请先设置DEEPSEEK_API_KEY环境变量")
    print("例如: set DEEPSEEK_API_KEY=你的API密钥")
    sys.exit(1)


def get_weather(city: str) -> str:
    """
    获取城市的天气
    """
    return f"{city}的天气是晴朗的"


my_agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather],
    system_prompt=f"""
    你是一个天气助手，你可以根据城市的名称来获取城市的天气。
    例如：
    {get_weather("北京")}
    {get_weather("上海")}
    """,
)

# 运行这个agent
inputs = {"messages": [{"role": "user", "content": "上海现在是什么天气？"}]}
try:
    for chunk in my_agent.stream(inputs, stream_mode="updates"):
        print(chunk)
except Exception as e:
    print(f"运行agent时出错: {e}")


