"""
第三课运行入口：Function Calling 工具调用体验

运行：python run_lesson3.py

试试这些问题：
  - 12345 * 67890 等于多少？
  - 现在几点了？
  - 北京天气怎么样？
  - 帮我查一下 AI Agent 是什么
"""
from agent.lesson3_function_calling import ToolAgent


def main():
    print("=" * 60)
    print("🔧 第三课：Function Calling 工具调用")
    print("核心：让 Agent 学会使用工具（计算器/时间/天气/搜索）")
    print("=" * 60)
    print("\n可用工具：")
    print("  🔢 calculator      - 数学计算")
    print("  🕐 get_current_time - 获取当前时间")
    print("  🌤️  get_weather     - 查询天气")
    print("  🔍 search_web      - 联网搜索")
    print("\n输入 /exit 退出\n")

    agent = ToolAgent("你是一个有用的 AI 助手。当用户的问题需要计算、查询时间、天气或搜索信息时，请使用工具。")

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

        print()
        reply = agent.run(user_input)
        print(f"\n🤖 Agent: {reply}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
