"""
第三课：Function Calling（工具调用）
核心：让 Agent 学会使用工具

工作流程：
  用户提问 -> LLM 决定是否调用工具 -> 执行工具 -> 结果返回 LLM -> LLM 生成最终回答

运行：python run_lesson3.py
"""
import json
from agent.llm_client import get_llm


# ============ 工具定义 ============
# 每个工具包含两部分：
#   1. 描述（TOOLS）：给 LLM 看的，说明工具用途和参数
#   2. 实现（函数）：真正执行的代码

def calculator(expression):
    """计算数学表达式"""
    try:
        # eval 仅限数字和运算符，生产环境需更安全的方案
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time():
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city):
    """获取城市天气（模拟数据，实际应调用天气 API）"""
    weather_db = {
        "北京": "晴，22°C，微风",
        "上海": "多云，25°C，东南风3级",
        "广州": "小雨，28°C，湿度85%",
        "深圳": "晴，27°C，紫外线强",
    }
    return weather_db.get(city, f"暂无{city}的天气数据")


def search_web(query):
    """联网搜索（模拟，实际应调用搜索 API）"""
    return f"搜索结果：关于'{query}'的模拟信息"


# 工具描述列表，传给 LLM 让它知道有哪些工具可用
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '100+200*2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 '北京'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "联网搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
]

# 工具名 -> 执行函数 的映射
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "search_web": search_web,
}


class ToolAgent:
    """支持工具调用的 Agent"""

    def __init__(self, system_prompt="", llm=None):
        self.system_prompt = system_prompt
        self.memory = []
        self.llm = llm or get_llm()

    def run(self, user_input, max_steps=5):
        """
        处理用户输入，支持多轮工具调用

        Args:
            user_input: 用户输入
            max_steps: 最大工具调用次数，防止无限循环
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory,
            {"role": "user", "content": user_input},
        ]

        for step in range(max_steps):
            print(f"  [步骤 {step + 1}] 调用 LLM...")

            # 调用 LLM，传入工具列表，让它决定是否调用工具
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls

            # LLM 不调用工具，直接返回回答
            if not tool_calls:
                reply = message.content
                self.memory.append({"role": "user", "content": user_input})
                self.memory.append({"role": "assistant", "content": reply})
                return reply

            print(f"  [步骤 {step + 1}] LLM 调用 {len(tool_calls)} 个工具")

            # 把 LLM 的工具调用请求加入消息历史
            messages.append(message)

            # 逐个执行工具
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"    调用工具: {tool_name}({tool_args})")

                # 执行对应的工具函数
                tool_func = TOOL_FUNCTIONS.get(tool_name)
                if tool_func:
                    tool_result = tool_func(**tool_args)
                else:
                    tool_result = f"错误：未知工具 {tool_name}"

                print(f"    工具返回: {tool_result}")

                # 把工具结果加入消息，让 LLM 基于结果继续回答
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result,
                })

        # 超过最大步数，强制让 LLM 直接回答
        print("  达到最大工具调用次数，强制生成回答")
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="none",
        )
        reply = response.choices[0].message.content
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    def clear_memory(self):
        self.memory = []
