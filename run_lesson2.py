"""
第二课运行入口：Prompt Engineering 对比演示

运行：python run_lesson2.py
"""
from agent.lesson2_prompt_engineering import compare_all, PromptEngineer


def main():
    print("=" * 60)
    print("🎓 第二课：Prompt Engineering 进阶技巧")
    print("=" * 60)
    print("\n本课程演示 6 种 Prompt 技巧对同一个问题的不同效果\n")

    # 默认演示问题
    question = "99+101等于几？"

    while True:
        print("\n" + "=" * 60)
        print("请选择：")
        print("  1 - 对比全部 6 种技巧（推荐先看这个）")
        print("  2 - 只看 Zero-shot")
        print("  3 - 只看 Few-shot")
        print("  4 - 只看 CoT 思维链")
        print("  5 - 只看角色扮演")
        print("  6 - 只看结构化输出")
        print("  7 - 只看分步指令")
        print("  8 - 换一个问题")
        print("  0 - 退出")
        print("=" * 60)

        choice = input("\n请输入选项: ").strip()

        if choice == "0":
            print("👋 再见！")
            break

        pe = PromptEngineer()

        if choice == "1":
            compare_all(question)
        elif choice == "2":
            print("\n" + pe.zero_shot(question))
        elif choice == "3":
            print("\n" + pe.few_shot(question))
        elif choice == "4":
            print("\n" + pe.cot(question))
        elif choice == "5":
            print("\n" + pe.role_play(question))
        elif choice == "6":
            print("\n" + pe.structured_output(question))
        elif choice == "7":
            print("\n" + pe.step_by_step(question))
        elif choice == "8":
            question = input("请输入新问题（如：15+27等于几？）: ").strip() or question
            print(f"问题已更新为：{question}")
        else:
            print("无效选项，请重新选择")

        input("\n按回车键继续...")


if __name__ == "__main__":
    main()
