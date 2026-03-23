import os
import sys
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek



# 添加项目的site-packages目录到Python路径
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Lib", "site-packages")
    )
)

# 加载环境变量
load_dotenv()

# 检查是否设置了OpenAI API密钥
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("请先设置DEEPSEEK_API_KEY环境变量")
    print("例如: set DEEPSEEK_API_KEY=你的API密钥")
    sys.exit(1)

def get_weather(city: str, temperature: str, **kwargs) -> str:
    """
    获取城市的天气
    """
    return f"{city}的天气是晴朗的，温度是{temperature}"


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "系统",
        f"""
    你是一个天气助手，你可以根据城市的名称来获取城市的天气。
    例如：
    {get_weather("北京","33°C")}
    {get_weather("上海","25°C")}
    """,
    ),
    ("用户", "上海现在是什么天气？"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
