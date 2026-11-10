"""
第一课运行入口：交互式对话体验

运行：
    python run_lesson1.py           # 使用真实大模型
    python run_lesson1.py --mock    # 使用模拟 LLM（无需 API）
"""
import sys
from agent.lesson1_simple_agent import SimpleAgent


def main():
    use_mock = "--mock" in sys.argv

    print("=" * 50)
    print("🤖 第一课：简单对话 Agent")
    print("核心：LLM（大脑） + 短期记忆")
    print(f"模式：{'模拟 LLM' if use_mock else '真实大模型'}")
    print("=" * 50)
    print("命令说明：")
    print("  直接输入文字  → 和 Agent 对话")
    print("  输入 /memory → 查看 Agent 的记忆")
    print("  输入 /clear  → 清空记忆")
    print("  输入 /exit   → 退出\n")

    # 创建 Agent
    if use_mock:
        agent = SimpleAgent("你是一个友好的 AI 助手，用简洁的方式回答用户问题。")
    else:
        # 使用真实大模型
        from agent.llm_client import get_llm
        llm = get_llm()
        agent = SimpleAgent(
            "你是一个友好的 AI 助手，用简洁的方式回答用户问题。",
            llm=llm
        )

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
            agent.show_memory()
            print()
            continue
        if user_input == "/clear":
            agent.clear_memory()
            print()
            continue

        reply = agent.generate_response(user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
