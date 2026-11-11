"""
第五课运行入口：ReAct 模式

用法：python run_lesson5.py
"""
from agent.lesson5_react import ReActAgent


def main():
    print("=" * 60)
    print("第五课：ReAct 模式")
    print("ReAct = Reasoning + Acting（思考 + 行动）")
    print("=" * 60)
    print()

    agent = ReActAgent()

    # 预设的演示问题
    demos = [
        "北京和上海的总人口是多少？",
        "深圳比广州少多少人？",
        "234 * 567 等于多少？",
        "北上广深四个城市的总人口是多少？",
    ]

    while True:
        print("\n请选择问题（输入序号）或直接输入问题：")
        for i, q in enumerate(demos, 1):
            print(f"  {i}. {q}")
        print("  0. 退出")

        choice = input("\n> ").strip()

        if choice == "0":
            print("再见")
            break

        if choice.isdigit() and 1 <= int(choice) <= len(demos):
            question = demos[int(choice) - 1]
        else:
            question = choice

        print(f"\n问题：{question}\n")
        answer = agent.run(question)
        print(f"\n最终答案：{answer}\n")


if __name__ == "__main__":
    main()
