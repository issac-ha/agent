"""
第三课运行入口：Function Calling 工具调用

用法：python run_lesson3.py

试试：12345 * 67890 等于多少？ / 现在几点了？ / 北京天气怎么样？
"""
from agent.lesson3_function_calling import ToolAgent


def main():
    print("=" * 60)
    print("第三课：Function Calling 工具调用")
    print("可用工具：计算器 / 时间 / 天气 / 搜索")
    print("=" * 60)
    print("输入 /exit 退出\n")

    agent = ToolAgent("你是一个有用的 AI 助手。需要计算、查询时间、天气或搜索时，请使用工具。")

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

        print()
        reply = agent.run(user_input)
        print(f"\nAgent: {reply}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
