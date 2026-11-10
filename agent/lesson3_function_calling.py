"""
第三课：Function Calling（工具调用）
==================================================
核心：让 Agent 学会使用工具，从"只能动口"变成"能动手"

工作流程：
  用户提问 → LLM 决定是否调用工具 → 执行工具 → 结果返回 LLM → LLM 生成最终回答

运行：python run_lesson3.py
"""
import json
from agent.llm_client import get_llm


# ============================================================
# 工具定义区：在这里定义 Agent 能用的工具
# 每个工具包含：名称、描述、参数 schema、执行函数
# ============================================================

def calculator(expression: str) -> str:
    """计算器：计算数学表达式"""
    try:
        # 用 eval 计算（仅限数字和运算符，生产环境需更安全的方案）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str) -> str:
    """获取城市天气（模拟数据，实际应调用天气 API）"""
    # 模拟数据
    weather_db = {
        "北京": "晴，22°C，微风",
        "上海": "多云，25°C，东南风3级",
        "广州": "小雨，28°C，湿度85%",
        "深圳": "晴，27°C，紫外线强",
    }
    return weather_db.get(city, f"暂无{city}的天气数据")


def search_web(query: str) -> str:
    """联网搜索（模拟，实际应调用搜索 API）"""
    return f"【搜索结果】关于'{query}'的信息：这是一个模拟的搜索结果，实际应调用搜索引擎API。"


# 工具注册表：把函数和它的描述绑定在一起
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除和括号。例如：'2+3*4'、'(10+5)/3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '100+200*2'"
                    }
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
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
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
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "联网搜索信息，用于回答需要实时或最新信息的问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
]

# 工具名称 → 执行函数 的映射
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "search_web": search_web,
}


# ============================================================
# 支持工具调用的 Agent
# ============================================================
class ToolAgent:
    """能使用工具的 Agent"""

    def __init__(self, system_prompt: str = "", llm=None):
        self.system_prompt = system_prompt
        self.memory = []
        self.llm = llm or get_llm()

    def run(self, user_input: str, max_steps: int = 5) -> str:
        """处理用户输入，支持多轮工具调用

        Args:
            user_input: 用户输入
            max_steps: 最大工具调用次数，防止无限循环

        Returns:
            最终回答
        """
        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory,
            {"role": "user", "content": user_input},
        ]

        for step in range(max_steps):
            print(f"  🔄 步骤 {step + 1}: 调用 LLM 思考...")

            # 调用 LLM，传入工具列表
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",  # 让 LLM 自己决定是否调用工具
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                # LLM 决定不调用工具，直接返回回答
                reply = message.content
                # 更新记忆
                self.memory.append({"role": "user", "content": user_input})
                self.memory.append({"role": "assistant", "content": reply})
                return reply

            # LLM 决定调用工具
            print(f"  🔧 LLM 决定调用 {len(tool_calls)} 个工具")

            # 把 LLM 的回复（含工具调用请求）加入消息历史
            messages.append(message)

            # 逐个执行工具
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"     🛠️  调用工具: {tool_name}({tool_args})")

                # 执行工具函数
                tool_func = TOOL_FUNCTIONS.get(tool_name)
                if tool_func:
                    tool_result = tool_func(**tool_args)
                else:
                    tool_result = f"错误：未知工具 {tool_name}"

                print(f"     📦 工具返回: {tool_result}")

                # 把工具结果加入消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result,
                })

        # 超过最大步数，强制让 LLM 回答
        print("  ⚠️ 达到最大工具调用次数，强制生成回答")
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="none",  # 强制不调用工具
        )
        reply = response.choices[0].message.content
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    def clear_memory(self):
        self.memory = []
