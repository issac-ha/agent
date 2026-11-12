"""第八课运行入口：多 Agent 协作系统"""
import sys
import os

def _fix_syspath():
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path and os.path.isdir(sp):
            sys.path.insert(0, sp)

_fix_syspath()

from agent.lesson8_multi_agent import HardwareConsultTeam


def main():
    print("=" * 60)
    print("第八课：多 Agent 协作系统")
    print("场景：硬件选购咨询团队（Planner + Researcher + Advisor）")
    print("输入需求，输入 /exit 退出")
    print("=" * 60)

    team = HardwareConsultTeam()

    demos = [
        "预算 8000 元，玩大型 3A 游戏，显卡和内存怎么选？",
        "预算 3000 元买显卡，主要做 AI 推理，有什么推荐？",
        "老电脑升级内存，DDR4 平台，预算 500 元以内",
    ]

    while True:
        print("\n可选示例（输入序号）或自定义需求：")
        for i, d in enumerate(demos, 1):
            print(f"  {i}. {d}")
        print("  0. 退出")

        choice = input("\n> ").strip()
        if choice == "0" or choice == "/exit":
            print("再见")
            break

        if choice.isdigit() and 1 <= int(choice) <= len(demos):
            request = demos[int(choice) - 1]
        else:
            request = choice

        if not request:
            continue

        team.consult(request)


if __name__ == "__main__":
    main()
