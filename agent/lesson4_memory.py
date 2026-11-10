"""
第四课：Memory 记忆系统
核心：
  1. 短期记忆 - 滑动窗口 + 对话摘要
  2. 长期记忆 - 向量存储关键信息，永久保存

运行：python run_lesson4.py
"""
import json
import math
import os
from agent.llm_client import get_llm


class ShortTermMemory:
    """
    短期记忆：滑动窗口 + 摘要

    策略：
      - 保留最近 N 轮对话（滑动窗口）
      - 超过阈值时，把老对话交给 LLM 摘要
      - 最终传给 LLM 的是：摘要 + 最近 N 轮
    """

    def __init__(self, llm, max_rounds=10, summary_threshold=20):
        self.llm = llm
        self.max_rounds = max_rounds            # 滑动窗口大小
        self.summary_threshold = summary_threshold  # 超过多少轮触发摘要
        self.messages = []                       # 原始对话历史
        self.summary = ""                        # 摘要内容

    def add(self, role, content):
        """添加一条消息"""
        self.messages.append({"role": role, "content": content})

        # 用户消息超过阈值时，触发摘要
        user_msg_count = len([m for m in self.messages if m["role"] == "user"])
        if user_msg_count > self.summary_threshold:
            self._summarize()

    def _summarize(self):
        """把老的对话总结成摘要，只保留最近的对话"""
        if len(self.messages) <= self.max_rounds:
            return

        old_messages = self.messages[:-self.max_rounds]
        recent_messages = self.messages[-self.max_rounds:]

        # 让 LLM 总结老对话的关键信息
        old_text = "\n".join([f"[{m['role']}]: {m['content']}" for m in old_messages])
        prompt = f"""请用简短的语言总结以下对话的关键信息，包括：
- 用户的身份、偏好、重要事实
- 讨论过的关键话题
- 已达成的结论

对话内容：
{old_text}

摘要（不超过200字）："""

        self.summary = self.llm.chat([{"role": "user", "content": prompt}])
        # 老对话已被摘要，只保留最近的
        self.messages = recent_messages

    def get_context(self):
        """获取当前要传给 LLM 的消息列表：摘要 + 最近对话"""
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"[对话摘要] {self.summary}"})
        context.extend(self.messages)
        return context

    def clear(self):
        self.messages = []
        self.summary = ""


class LongTermMemory:
    """
    长期记忆：用向量存储关键信息

    原理：
      1. 把每条记忆转成向量
      2. 用余弦相似度检索相关记忆
      3. 把相关记忆注入对话上下文

    注意：这是简化实现，实际项目用 Chroma / Milvus 等向量数据库
    """

    def __init__(self, storage_file="long_term_memory.json"):
        self.storage_file = storage_file
        self.memories = []  # [{content, vector}]
        self._load()

    def _text_to_vector(self, text):
        """
        文本转向量（简化版：字符 bigram）
        真实项目用 Embedding 模型（如 BGE、text-embedding-3）
        """
        # 用相邻两字作为特征，适合中文
        text = text.replace(" ", "").replace("\n", "")
        bigrams = [text[i:i+2] for i in range(len(text) - 1)]
        word_count = {}
        for bg in bigrams:
            word_count[bg] = word_count.get(bg, 0) + 1
        return word_count

    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度，值越接近 1 越相关"""
        all_words = set(vec1.keys()) | set(vec2.keys())
        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in all_words)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)

    def add(self, content):
        """添加一条长期记忆"""
        vector = self._text_to_vector(content)
        self.memories.append({"content": content, "vector": vector})
        self._save()

    def search(self, query, top_k=3):
        """搜索与查询最相关的 top_k 条记忆"""
        query_vec = self._text_to_vector(query)
        scored = []
        for mem in self.memories:
            score = self._cosine_similarity(query_vec, mem["vector"])
            scored.append((score, mem["content"]))
        scored.sort(reverse=True)
        return [content for score, content in scored[:top_k] if score > 0]

    def _save(self):
        """保存记忆到文件（只存文本，向量可重新计算）"""
        data = [m["content"] for m in self.memories]
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载记忆"""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.memories = [{"content": c, "vector": self._text_to_vector(c)} for c in data]

    def clear(self):
        self.memories = []
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)


class MemoryAgent:
    """整合短期 + 长期记忆的 Agent"""

    def __init__(self, system_prompt="", llm=None):
        self.system_prompt = system_prompt
        self.llm = llm or get_llm()
        self.short_term = ShortTermMemory(self.llm)
        self.long_term = LongTermMemory()

    def run(self, user_input):
        # 1. 从长期记忆中检索相关信息
        relevant_memories = self.long_term.search(user_input)

        # 2. 把相关记忆注入系统提示词
        system_content = self.system_prompt
        if relevant_memories:
            system_content += "\n\n[相关历史记忆]\n" + "\n".join(relevant_memories)

        # 3. 构建完整消息：系统提示 + 短期记忆 + 当前输入
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.short_term.get_context())
        messages.append({"role": "user", "content": user_input})

        # 4. 调用 LLM
        reply = self.llm.chat(messages)

        # 5. 更新短期记忆
        self.short_term.add("user", user_input)
        self.short_term.add("assistant", reply)

        return reply

    def remember(self, info):
        """把重要信息存入长期记忆"""
        self.long_term.add(info)
        print(f"已记住：{info}")

    def show_long_term(self):
        """打印所有长期记忆"""
        print("长期记忆列表：")
        for i, mem in enumerate(self.long_term.memories, 1):
            print(f"  {i}. {mem['content']}")
