from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b

@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。"""
    # 此处可接入真实天气API
    return f"{city}的天气是晴天，25度。"

tools = [multiply, get_weather]
