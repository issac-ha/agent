"""
第四课运行入口：Memory 记忆系统体验

运行：python run_lesson4.py

试试这些操作：
  直接输入 → 和 Agent 对话
  /remember 我喜欢用 Python 编程 → 存入长期记忆
  /memory → 查看长期记忆
  /clear → 清空记忆
  /exit → 退出
"""
from agent.lesson4_memory import MemoryAgent


def main():
    print("=" * 60)
    print("🧠 第四课：Memory 记忆系统")
    print("核心：短期记忆（滑动窗口+摘要） + 长期记忆（向量存储）")
    print("=" * 60)
    print("\n命令说明：")
    print("  直接输入        → 对话")
    print("  /remember 内容  → 把重要信息存入长期记忆")
    print("  /memory         → 查看所有长期记忆")
    print("  /clear          → 清空记忆")
    print("  /exit           → 退出\n")

    agent = MemoryAgent("你是一个友好的 AI 助手。")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("👋 再见！")
            break
        if user_input == "/memory":
            agent.show_long_term()
            print()
            continue
        if user_input == "/clear":
            agent.short_term.clear()
            agent.long_term.clear()
            print("🧹 所有记忆已清空\n")
            continue
        if user_input.startswith("/remember "):
            info = user_input[len("/remember "):]
            agent.remember(info)
            print()
            continue

        reply = agent.run(user_input)
        print(f"🤖 Agent: {reply}\n")


if __name__ == "__main__":
    main()
