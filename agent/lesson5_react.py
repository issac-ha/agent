"""
第五课：ReAct 模式
核心：Reasoning（推理） + Acting（行动）

ReAct 的工作流程：
  Thought（思考） -> Action（行动） -> Observation（观察） -> Thought -> ... -> Final Answer

运行：python run_lesson5.py
"""
import re
import json
from agent.llm_client import get_llm


# ============ 工具定义 ============

def search(query):
    """搜索（模拟，返回固定信息）"""
    db = {
        "北京人口": "北京常住人口约 2184 万",
        "上海人口": "上海常住人口约 2487 万",
        "广州人口": "广州常住人口约 1881 万",
        "深圳人口": "深圳常住人口约 1766 万",
    }
    return db.get(query, f"未找到'{query}'的相关信息")


def calculator(expression):
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


# 工具描述
TOOLS = {
    "search": {
        "name": "search",
        "description": "搜索事实信息，如人口、面积等",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    },
    "calculator": {
        "name": "calculator",
        "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    },
}

TOOL_FUNCTIONS = {
    "search": search,
    "calculator": calculator,
}

# ReAct 系统提示词，定义 Agent 的思考格式
REACT_PROMPT = """你是一个会思考和使用工具的 AI 助手。

你必须严格按以下步骤工作，每次只能输出一步：

第一步：思考，输出
Thought: <你的思考>

第二步：决定是否需要工具。如果需要，输出
Action: <工具名>
Action Input: <JSON格式参数>

然后停止，等待工具返回结果。

工具返回结果后（你会收到 Observation: ...），继续思考：
- 如果信息足够，输出：Thought: <总结> 然后 Final Answer: <答案>
- 如果还需要更多信息，输出新的 Thought 和 Action

可用工具：
- search: 搜索事实信息（人口、面积等），参数：{{"query": "关键词"}}
- calculator: 计算数学表达式，参数：{{"expression": "算式"}}

重要规则：
- 每次回复只能包含一个 Thought 和一个 Action，不要自己编造 Observation
- 调用 search 时，query 要精确，如"北京人口"而不是"北京上海人口"
- 最终答案必须以 Final Answer: 开头

问题：{question}

开始。"""


class ReActAgent:
    """基于 ReAct 模式的 Agent"""

    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.max_steps = 10

    def run(self, question):
        """
        ReAct 循环：思考 -> 行动 -> 观察 -> 思考 -> ... -> 最终答案
        """
        # 初始消息
        messages = [{"role": "user", "content": REACT_PROMPT.format(question=question)}]

        for step in range(self.max_steps):
            print(f"\n  [步骤 {step + 1}] LLM 思考中...")

            # 调用 LLM
            response = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"  {response}")

            # 检查是否已经给出最终答案
            if "Final Answer:" in response:
                final = response.split("Final Answer:")[-1].strip()
                return final

            # 解析 Action 和 Action Input
            action_match = re.search(r"Action:\s*(.+)", response)
            input_match = re.search(r"Action Input:\s*(\{.+?\}|.+)", response)

            if not action_match or not input_match:
                print("  [警告] 无法解析 Action，尝试让 LLM 继续")
                messages.append({"role": "user", "content": "请按照格式输出 Action 和 Action Input"})
                continue

            action_name = action_match.group(1).strip()
            action_input_raw = input_match.group(1).strip()

            # 解析参数（尝试 JSON，失败则当字符串）
            try:
                action_input = json.loads(action_input_raw)
            except json.JSONDecodeError:
                action_input = {"query": action_input_raw}

            print(f"  [调用工具] {action_name}({action_input})")

            # 执行工具
            tool_func = TOOL_FUNCTIONS.get(action_name)
            if tool_func:
                observation = tool_func(**action_input)
            else:
                observation = f"错误：未知工具 {action_name}"

            print(f"  [工具返回] {observation}")

            # 把观察结果发给 LLM
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # 超过最大步数
        return "思考步数超限，未能得出最终答案"


def main():
    agent = ReActAgent()

    print("=" * 60)
    print("第五课：ReAct 模式")
    print("ReAct = Reasoning + Acting（思考 + 行动）")
    print("=" * 60)

    # 演示一个需要多步推理的问题
    question = "北京和上海的总人口是多少？"
    print(f"\n问题：{question}\n")

    answer = agent.run(question)
    print(f"\n{'='*60}")
    print(f"最终答案：{answer}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
