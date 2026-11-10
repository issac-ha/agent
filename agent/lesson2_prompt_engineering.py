"""
第二课：Prompt Engineering 进阶技巧
==================================================
6 大核心技巧：
  1. Zero-shot（零样本）—— 直接给指令
  2. Few-shot（少样本）—— 给几个例子
  3. CoT 思维链 —— 让模型一步步想
  4. 角色扮演 —— 设定专家身份
  5. 结构化输出 —— 指定返回格式
  6. 分步指令 —— 把任务拆成步骤

运行：python run_lesson2.py
"""


class PromptEngineer:
    """演示不同 Prompt 技巧的效果差异"""

    def __init__(self):
        # 模拟一个"知识库"，让 Few-shot 和 CoT 有东西可学
        self.knowledge = {
            "1+1": "2",
            "15+27": "42",
            "99+101": "200",
        }

    # ============================================================
    # 技巧 1：Zero-shot（零样本）
    # ============================================================
    def zero_shot(self, question: str) -> str:
        """直接给指令，不给例子"""
        prompt = f"请回答以下问题：\n{question}"
        return self._mock_llm(prompt, mode="zero_shot")

    # ============================================================
    # 技巧 2：Few-shot（少样本）
    # ============================================================
    def few_shot(self, question: str) -> str:
        """给几个例子，模型照葫芦画瓢"""
        prompt = f"""请按照示例回答问题：

示例1：
问题：1+1等于几？
回答：2

示例2：
问题：15+27等于几？
回答：42

现在请回答：
问题：{question}
回答："""
        return self._mock_llm(prompt, mode="few_shot")

    # ============================================================
    # 技巧 3：CoT（Chain of Thought 思维链）
    # ============================================================
    def cot(self, question: str) -> str:
        """让模型一步步推理，而不是直接给答案"""
        prompt = f"""请一步步思考后回答问题：

问题：{question}

请按以下格式回答：
思考过程：<写出你的计算步骤>
最终答案：<结果>"""
        return self._mock_llm(prompt, mode="cot")

    # ============================================================
    # 技巧 4：角色扮演
    # ============================================================
    def role_play(self, question: str) -> str:
        """设定一个专家身份，让模型用专业视角回答"""
        prompt = f"""你是一位拥有 10 年经验的资深数学老师，擅长用通俗的语言讲解数学题。

问题：{question}

请用老师的口吻回答："""
        return self._mock_llm(prompt, mode="role_play")

    # ============================================================
    # 技巧 5：结构化输出
    # ============================================================
    def structured_output(self, question: str) -> str:
        """指定返回格式（JSON / 表格 / 列表），方便程序解析"""
        prompt = f"""请回答以下问题，并严格按照 JSON 格式输出：

问题：{question}

输出格式：
{{
  "answer": "最终答案",
  "confidence": "置信度 high/medium/low",
  "explanation": "简短解释"
}}

只输出 JSON，不要其他内容。"""
        return self._mock_llm(prompt, mode="structured")

    # ============================================================
    # 技巧 6：分步指令
    # ============================================================
    def step_by_step(self, question: str) -> str:
        """把复杂任务拆成多个步骤，逐步引导"""
        prompt = f"""请按照以下步骤处理用户请求：

步骤1：理解问题
步骤2：列出已知条件
步骤3：选择合适的方法
步骤4：计算结果
步骤5：检查答案

问题：{question}

请依次完成每个步骤："""
        return self._mock_llm(prompt, mode="step_by_step")

    # ============================================================
    # 模拟 LLM
    # ============================================================
    def _mock_llm(self, prompt: str, mode: str) -> str:
        """模拟大模型对不同 Prompt 的响应差异

        真实项目中替换为：
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        """
        # 提取问题（简化处理），并清理掉"等于几"、"？"等干扰文字
        q = prompt.split("问题：")[-1].split("\n")[0].strip() if "问题：" in prompt else prompt
        q_clean = q.replace("等于几", "").replace("等于多少", "").replace("？", "").replace("?", "").strip()

        if mode == "zero_shot":
            # Zero-shot 容易答非所问或太简略
            # 注意：没有例子引导时，模型可能直接说不知道
            return self.knowledge.get(q_clean, "我不知道")

        elif mode == "few_shot":
            # Few-shot 因为有例子，准确率高
            return self.knowledge.get(q_clean, "根据示例的规律，答案应该是...")

        elif mode == "cot":
            # CoT 会展示推理过程
            if "+" in q_clean:
                a, b = q_clean.split("+")
                a, b = int(a), int(b)
                return f"""思考过程：
  第一步：把 {a} 拆成 {a//10}个十 和 {a%10}个一
  第二步：把 {b} 拆成 {b//10}个十 和 {b%10}个一
  第三步：十位相加 {a//10} + {b//10} = {(a//10)+(b//10)}
  第四步：个位相加 {a%10} + {b%10} = {(a%10)+(b%10)}
  第五步：合并得到 {(a//10)+(b//10)}个十 和 {(a%10)+(b%10)}个一
最终答案：{a + b}"""
            return "思考过程：...\n最终答案：..."

        elif mode == "role_play":
            # 角色扮演会用老师的口吻
            return f"同学你好！这道题其实很简单哦～ 我们把 {q_clean} 算一下，答案就是 {self.knowledge.get(q_clean, self.knowledge.get(q, '...'))}。明白了吗？还有问题随时问老师！"

        elif mode == "structured":
            # 结构化输出严格返回 JSON
            import json
            return json.dumps({
                "answer": self.knowledge.get(q, "未知"),
                "confidence": "high",
                "explanation": "根据加法运算规则得出"
            }, ensure_ascii=False, indent=2)

        elif mode == "step_by_step":
            # 分步指令会逐步完成
            if "+" in q_clean:
                a, b = q_clean.split("+")
                a, b = int(a), int(b)
                return f"""步骤1【理解问题】：用户想知道 {a} + {b} 的结果
步骤2【已知条件】：两个加数分别是 {a} 和 {b}
步骤3【选择方法】：使用加法运算
步骤4【计算结果】：{a} + {b} = {a + b}
步骤5【检查答案】：用 {a + b} - {a} = {b} ✓ 答案正确
最终结果：{a + b}"""
            return "步骤1：...\n步骤2：...\n..."

        return "无法处理"


# ============================================================
# 对比演示
# ============================================================
def compare_all(question: str = "99+101等于几？"):
    """对比同一种问题，用不同 Prompt 技巧的效果"""
    pe = PromptEngineer()

    print(f"\n{'='*60}")
    print(f"📝 问题：{question}")
    print(f"{'='*60}\n")

    techniques = [
        ("1️⃣ Zero-shot（零样本）", pe.zero_shot),
        ("2️⃣ Few-shot（少样本）", pe.few_shot),
        ("3️⃣ CoT（思维链）", pe.cot),
        ("4️⃣ 角色扮演", pe.role_play),
        ("5️⃣ 结构化输出", pe.structured_output),
        ("6️⃣ 分步指令", pe.step_by_step),
    ]

    for name, func in techniques:
        print(f"\n{'─'*60}")
        print(f"▶ {name}")
        print(f"{'─'*60}")
        result = func(question)
        print(result)
