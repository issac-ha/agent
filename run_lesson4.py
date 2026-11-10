"""
第四课运行入口：Memory 记忆系统

用法：python run_lesson4.py

命令：
  直接输入        对话
  /remember 内容  存入长期记忆
  /memory         查看长期记忆
  /clear          清空记忆
  /exit           退出
"""
from agent.lesson4_memory import MemoryAgent


def main():
    print("=" * 60)
    print("第四课：Memory 记忆系统")
    print("短期记忆（滑动窗口+摘要） + 长期记忆（向量存储）")
    print("=" * 60)
    print()

    agent = MemoryAgent("你是一个友好的 AI 助手。")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("再见")
            break
        if user_input == "/memory":
            agent.show_long_term()
            print()
            continue
        if user_input == "/clear":
            agent.short_term.clear()
            agent.long_term.clear()
            print("所有记忆已清空\n")
            continue
        if user_input.startswith("/remember "):
            info = user_input[len("/remember "):]
            agent.remember(info)
            print()
            continue

        reply = agent.run(user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
