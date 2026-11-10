"""
第二课运行入口：Prompt Engineering 对比演示

用法：python run_lesson2.py
"""
from agent.lesson2_prompt_engineering import compare_all, PromptEngineer


def main():
    print("=" * 60)
    print("第二课：Prompt Engineering 进阶技巧")
    print("=" * 60)
    print("演示 6 种 Prompt 技巧对同一问题的不同效果\n")

    question = "99+101等于几？"

    while True:
        print("\n" + "=" * 60)
        print("请选择：")
        print("  1 - 对比全部 6 种技巧")
        print("  2 - Zero-shot")
        print("  3 - Few-shot")
        print("  4 - CoT 思维链")
        print("  5 - 角色扮演")
        print("  6 - 结构化输出")
        print("  7 - 分步指令")
        print("  8 - 换一个问题")
        print("  0 - 退出")
        print("=" * 60)

        choice = input("\n请输入选项: ").strip()

        if choice == "0":
            print("再见")
            break

        pe = PromptEngineer()

        if choice == "1":
            compare_all(question)
        elif choice in ("2", "3", "4", "5", "6", "7"):
            methods = {
                "2": pe.zero_shot,
                "3": pe.few_shot,
                "4": pe.cot,
                "5": pe.role_play,
                "6": pe.structured_output,
                "7": pe.step_by_step,
            }
            print("\n" + methods[choice](question))
        elif choice == "8":
            question = input("请输入新问题: ").strip() or question
            print(f"问题已更新为：{question}")
        else:
            print("无效选项")

        input("\n按回车继续...")


if __name__ == "__main__":
    main()
