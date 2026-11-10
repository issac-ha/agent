"""
第一课运行入口

用法：
    python run_lesson1.py           使用真实大模型
    python run_lesson1.py --mock    使用模拟模式（无需 API）
"""
import sys
from agent.lesson1_simple_agent import SimpleAgent


def main():
    use_mock = "--mock" in sys.argv

    print("=" * 50)
    print("第一课：简单对话 Agent")
    print("核心：LLM + 短期记忆")
    print(f"模式：{'模拟 LLM' if use_mock else '真实大模型'}")
    print("=" * 50)
    print("命令：直接输入对话 | /memory 查看记忆 | /clear 清空 | /exit 退出")
    print()

    if use_mock:
        agent = SimpleAgent("你是一个友好的 AI 助手。")
    else:
        from agent.llm_client import get_llm
        agent = SimpleAgent("你是一个友好的 AI 助手。", llm=get_llm())

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
