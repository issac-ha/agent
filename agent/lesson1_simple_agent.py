"""
第一课：最简单的 AI Agent —— 对话机器人
核心：LLM（大脑） + 短期记忆（对话历史）

运行方式：
    python run_lesson1.py          # 使用真实大模型
    python run_lesson1.py --mock   # 使用模拟 LLM（无需 API）
"""


class SimpleAgent:
    """一个简化版的 Agent 类

    支持两种模式：
      1. 真实模式：传入 llm 参数，调用真实大模型 API
      2. 模拟模式：不传 llm，用内置模拟逻辑（方便无 API 时学习）
    """

    def __init__(self, system_prompt: str = "", llm=None):
        # 系统提示词：定义 Agent 的身份和行为
        self.system_prompt = system_prompt
        # 短期记忆：存储对话历史
        self.memory = []
        # LLM 客户端（可选，传入则用真实模型）
        self.llm = llm

    def generate_response(self, user_input: str) -> str:
        """核心方法：接收用户输入，返回 Agent 回复"""
        # 1. 构建完整的消息列表（系统提示 + 历史对话 + 当前输入）
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory,
            {"role": "user", "content": user_input},
        ]

        # 2. 调用 LLM
        if self.llm:
            # 真实模式：调用大模型 API
            reply = self.llm.chat(messages)
        else:
            # 模拟模式：用内置逻辑（无需 API Key）
            reply = self._mock_llm(messages)

        # 3. 更新记忆：把用户输入和 Agent 回复都存起来
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": reply})

        return reply

    def _mock_llm(self, messages: list) -> str:
        """模拟 LLM 响应（无需 API Key 时使用）"""
        last_user_msg = messages[-1]["content"]
        history_count = len([m for m in self.memory if m["role"] == "user"])

        if "你好" in last_user_msg or "hi" in last_user_msg.lower():
            return f"你好！我是你的 AI 助手 🤖 这是我们的第 {history_count + 1} 轮对话。有什么可以帮你的？"
        if "名字" in last_user_msg:
            return "我叫 SimBot，是一个用 Python 写的简单 Agent 示例。"
        if "能做什么" in last_user_msg or "功能" in last_user_msg:
            return "我目前能陪你聊天，记住我们的对话内容。后续课程我会学会使用工具、做规划哦！"
        if "天气" in last_user_msg:
            return "抱歉，我现在还没有联网搜索的工具能力。在第三课我们会学习让 Agent 调用工具！"
        return f'你说的是"{last_user_msg}"。我记住了我们之前聊过 {history_count} 轮。继续问我吧～'

    def clear_memory(self):
        """清空记忆"""
        self.memory = []
        print("🧹 记忆已清空")

    def show_memory(self):
        """查看记忆（调试用）"""
        print("📝 当前对话记忆：")
        for i, m in enumerate(self.memory, 1):
            print(f"  {i}. [{m['role']}]: {m['content']}")
