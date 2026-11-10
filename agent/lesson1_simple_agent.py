"""
第一课：简单对话 Agent
核心：LLM + 短期记忆

用法：
    python run_lesson1.py            使用真实大模型
    python run_lesson1.py --mock     使用模拟模式（无需 API）
"""


class SimpleAgent:
    """
    简单 Agent，支持两种模式：
    - 真实模式：传入 llm 参数，调用大模型 API
    - 模拟模式：不传 llm，用内置逻辑（无 API 时学习用）
    """

    def __init__(self, system_prompt="", llm=None):
        # 系统提示词，定义 Agent 的身份和行为
        self.system_prompt = system_prompt
        # 短期记忆，存储对话历史
        self.memory = []
        # LLM 客户端，可选
        self.llm = llm

    def generate_response(self, user_input):
        """接收用户输入，返回 Agent 回复"""
        # 组装消息：系统提示 + 历史对话 + 当前输入
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory,
            {"role": "user", "content": user_input},
        ]

        # 调用 LLM，有真实模型就用真实的，否则用模拟
        if self.llm:
            reply = self.llm.chat(messages)
        else:
            reply = self._mock_llm(messages)

        # 更新记忆，把这轮对话存起来
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    def _mock_llm(self, messages):
        """模拟 LLM 响应，无需 API Key"""
        msg = messages[-1]["content"]
        # 统计历史对话轮数，演示 Agent 有记忆
        count = len([m for m in self.memory if m["role"] == "user"])

        if "你好" in msg or "hi" in msg.lower():
            return f"你好，第 {count + 1} 轮对话，有什么可以帮你？"
        if "名字" in msg:
            return "我叫 SimBot。"
        if "能做什么" in msg or "功能" in msg:
            return "我能陪你聊天，记住对话内容。"
        if "天气" in msg:
            return "我还没有联网能力。"
        return f'你说的是"{msg}"，已聊 {count} 轮。'

    def clear_memory(self):
        """清空记忆"""
        self.memory = []
        print("记忆已清空")

    def show_memory(self):
        """打印当前对话记忆，调试用"""
        print("对话记忆：")
        for i, m in enumerate(self.memory, 1):
            print(f"  {i}. [{m['role']}] {m['content']}")
