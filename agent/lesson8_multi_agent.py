"""
第八课：多 Agent 协作系统

核心：多个 Agent 分工合作，每个 Agent 负责一类任务

本课场景：硬件选购咨询团队
  Planner Agent   - 分析用户需求，拆解成子任务
  Researcher Agent - 查询硬件资料（用第七课的 RAG）
  Advisor Agent   - 综合资料，给出购买建议

运行：python run_lesson8.py
"""
import sys
import os

def _fix_syspath():
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path and os.path.isdir(sp):
            sys.path.insert(0, sp)

_fix_syspath()

from agent.llm_client import get_llm
from agent.lesson7_rag import build_vector_store


# ============ Agent 基类 ============

class BaseAgent:
    """所有 Agent 的基类：有自己的角色提示词和 LLM"""

    def __init__(self, name, role_prompt, llm=None):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm or get_llm()

    def think(self, task):
        """Agent 处理任务，返回结果"""
        messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": task},
        ]
        result = self.llm.chat(messages)
        print(f"\n[{self.name}] 输出：\n{result}\n")
        return result


# ============ 三个角色 Agent ============

class PlannerAgent(BaseAgent):
    """规划 Agent：分析用户需求，拆解任务"""

    def __init__(self, llm=None):
        super().__init__(
            name="Planner",
            role_prompt="""你是一个需求分析专家。
用户会告诉你他的硬件选购需求，你需要：
1. 提取关键信息：预算、用途、关注的硬件类型（显卡/内存）
2. 把需求拆成 2-3 个具体的查询子任务

输出格式（严格遵守）：
预算：<金额>元
用途：<游戏/办公/AI/其他>
硬件类型：<显卡/内存>
子任务：
1. <查询任务1>
2. <查询任务2>""",
            llm=llm,
        )

    def plan(self, user_request):
        print(f"[Planner] 收到需求：{user_request}")
        return self.think(user_request)


class ResearcherAgent(BaseAgent):
    """研究 Agent：根据子任务查询硬件资料"""

    def __init__(self, store, llm=None):
        super().__init__(
            name="Researcher",
            role_prompt="""你是一个硬件资料查询助手。
你会收到查询任务和检索到的参考资料。
请把参考资料整理成简洁的要点，保留价格、型号、参数等关键信息。""",
            llm=llm,
        )
        self.store = store  # 向量数据库

    def research(self, plan_result):
        # 从规划结果中提取子任务（简单按行解析）
        subtasks = [
            line.split(".", 1)[-1].strip()
            for line in plan_result.split("\n")
            if line.strip() and line.strip()[0].isdigit() and "." in line
        ]

        all_findings = []
        for task in subtasks:
            print(f"\n[Researcher] 查询子任务：{task}")
            # 用 RAG 检索相关文档
            results = self.store.search(task, top_k=2)
            for r in results:
                print(f"  找到资料 (相似度 {r['score']:.4f})：{r['text'][:50]}...")
                all_findings.append(r["text"])

        # 让 LLM 整理资料
        findings_text = "\n".join(all_findings) if all_findings else "未找到相关资料"
        return self.think(f"查询任务：{subtasks}\n\n检索到的资料：\n{findings_text}")


class AdvisorAgent(BaseAgent):
    """顾问 Agent：综合所有信息，给出购买建议"""

    def __init__(self, llm=None):
        super().__init__(
            name="Advisor",
            role_prompt="""你是一个资深硬件选购顾问。
你会收到用户的需求分析和查询到的硬件资料。
请给出 1-2 个具体的购买建议，包含：
- 推荐型号和理由
- 价格是否符合预算
- 有什么需要注意的地方
回答要实用、接地气，像朋友给建议一样。""",
            llm=llm,
        )

    def advise(self, plan_result, research_result):
        task = f"需求分析：\n{plan_result}\n\n查询到的资料：\n{research_result}"
        return self.think(task)


# ============ 多 Agent 协作团队 ============

class HardwareConsultTeam:
    """
    硬件选购咨询团队
    流水线式协作：Planner -> Researcher -> Advisor
    """

    def __init__(self, llm=None):
        if llm is None:
            llm = get_llm()
        self.store = build_vector_store()
        self.planner = PlannerAgent(llm)
        self.researcher = ResearcherAgent(self.store, llm)
        self.advisor = AdvisorAgent(llm)

    def consult(self, user_request):
        """完整的多 Agent 协作流程"""
        print("=" * 60)
        print(f"用户需求：{user_request}")
        print("=" * 60)

        # 步骤 1：规划 Agent 拆解任务
        print("\n>>> 步骤 1：Planner 分析需求")
        plan = self.planner.plan(user_request)

        # 步骤 2：研究 Agent 查询资料
        print("\n>>> 步骤 2：Researcher 查询资料")
        research = self.researcher.research(plan)

        # 步骤 3：顾问 Agent 给出建议
        print("\n>>> 步骤 3：Advisor 给出建议")
        advice = self.advisor.advise(plan, research)

        print("\n" + "=" * 60)
        print("最终建议：")
        print(advice)
        print("=" * 60)
        return advice


# ============ 主入口 ============

def main():
    team = HardwareConsultTeam()
    team.consult("我想配一台玩游戏的电脑，预算 8000 元左右，主要玩大型 3A 游戏，显卡和内存怎么选？")


if __name__ == "__main__":
    main()
