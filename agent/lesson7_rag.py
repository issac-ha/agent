"""
第七课：RAG 检索增强生成

RAG 完整流程：
  1. Load - 加载文档
  2. Split - 文档分块
  3. Embed - 向量化
  4. Store - 存入向量数据库
  5. Retrieve - 检索相关文档（余弦相似度）
  6. Generate - 拼进 Prompt，LLM 生成回答

运行：python run_lesson7.py
"""
import sys
import os
import math

def _fix_syspath():
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path and os.path.isdir(sp):
            sys.path.insert(0, sp)

_fix_syspath()

from dotenv import load_dotenv
from agent.llm_client import get_llm

load_dotenv()


# ============ 硬件知识库（模拟从文档中提取的内容） ============

HARDWARE_DOCS = [
    {
        "id": "gpu_rtx5090",
        "text": "NVIDIA GeForce RTX 5090 是当前最强消费级显卡，采用 Blackwell 架构，拥有 21760 个 CUDA 核心，32GB GDDR7 显存，首发价格 19999 元。性能比 RTX 4090 提升约 35%，适合 4K 游戏和 AI 推理。",
        "category": "显卡"
    },
    {
        "id": "gpu_rtx5080",
        "text": "NVIDIA GeForce RTX 5080 采用 Blackwell 架构，10752 个 CUDA 核心，16GB GDDR7 显存，首发价格 8299 元。性能接近 RTX 4090，功耗更低，是高端游戏的性价比之选。",
        "category": "显卡"
    },
    {
        "id": "gpu_rtx5070ti",
        "text": "NVIDIA GeForce RTX 5070 Ti 拥有 8960 个 CUDA 核心，16GB GDDR7 显存，首发价格 6299 元。主打 2K 高刷游戏，性能超越 RTX 4080。",
        "category": "显卡"
    },
    {
        "id": "gpu_rtx4060",
        "text": "NVIDIA GeForce RTX 4060 采用 Ada Lovelace 架构，3072 个 CUDA 核心，8GB GDDR6 显存，当前价格约 2399 元。适合 1080P 游戏和入门级 AI 应用。",
        "category": "显卡"
    },
    {
        "id": "gpu_rx7800xt",
        "text": "AMD Radeon RX 7800 XT 拥有 3840 个流处理器，16GB GDDR6 显存，当前价格约 3699 元。性能介于 RTX 4070 和 RTX 4070 Ti 之间，显存更大。",
        "category": "显卡"
    },
    {
        "id": "ram_ddr5_6000",
        "text": "DDR5 6000MHz 内存条是当前主流选择，16GB 套装价格约 350 元，32GB 套装约 650 元。频率比 DDR4 提升 50%，是新装机首选。",
        "category": "内存"
    },
    {
        "id": "ram_ddr5_6400",
        "text": "DDR5 6400MHz 内存条属于高频款，16GB 套装价格约 450 元，32GB 套装约 850 元。适合追求极致性能的游戏玩家和内容创作者。",
        "category": "内存"
    },
    {
        "id": "ram_ddr4_3200",
        "text": "DDR4 3200MHz 内存条是上代主流，16GB 套装价格约 200 元，32GB 套装约 380 元。虽然性能不如 DDR5，但价格便宜，适合老平台升级。",
        "category": "内存"
    },
    {
        "id": "ram_ddr5_7200",
        "text": "DDR5 7200MHz 是目前民用最高频内存，16GB 套装价格约 800 元，32GB 套装约 1500 元。需要搭配高端主板才能稳定运行。",
        "category": "内存"
    },
]


# ============ Embedding：文本转向量（基于词频，归一化） ============

def text_to_vector(text):
    """
    文本转向量：用字符 bigram 词频，并做 L2 归一化

    归一化后，余弦相似度 = 向量点积，计算更简单
    真实项目用 OpenAI text-embedding 或 BGE 模型
    """
    text = text.replace(" ", "").replace("\n", "")
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]

    word_count = {}
    for bg in bigrams:
        word_count[bg] = word_count.get(bg, 0) + 1

    # L2 归一化：让向量长度为 1
    norm = math.sqrt(sum(v ** 2 for v in word_count.values()))
    if norm == 0:
        return word_count
    return {k: v / norm for k, v in word_count.items()}


def cosine_similarity(vec1, vec2):
    """两个归一化向量的余弦相似度 = 点积"""
    common = set(vec1.keys()) & set(vec2.keys())
    return sum(vec1[k] * vec2[k] for k in common)


# ============ 向量数据库（自己实现，不用 chromadb） ============

class VectorStore:
    """
    简化版向量数据库

    真实项目用 Chroma / Milvus / Qdrant
    这里自己实现，方便理解原理
    """

    def __init__(self):
        self.docs = []      # [{id, text, metadata, vector}]

    def add(self, documents):
        """添加文档，自动向量化"""
        for doc in documents:
            self.docs.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "vector": text_to_vector(doc["text"]),
            })

    def search(self, query, top_k=3):
        """用余弦相似度检索最相关的文档"""
        query_vec = text_to_vector(query)

        scored = []
        for doc in self.docs:
            score = cosine_similarity(query_vec, doc["vector"])
            scored.append((score, doc))

        # 按相似度降序排列
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:top_k]:
            results.append({"text": doc["text"], "score": score, "id": doc["id"]})

        return results


# ============ RAG 完整流程 ============

def build_vector_store():
    """步骤 1-4：加载文档、向量化、存入向量数据库"""
    store = VectorStore()
    store.add(HARDWARE_DOCS)
    print(f"已存入 {len(store.docs)} 条文档到向量数据库\n")
    return store


def retrieve(store, query, top_k=3):
    """步骤 5：检索相关文档"""
    results = store.search(query, top_k)

    print(f"检索到 {len(results)} 条相关文档：")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] (相似度: {r['score']:.4f}) {r['text'][:60]}...")

    return [r["text"] for r in results]


def rag_answer(store, question, llm=None):
    """步骤 5-6：检索 + 生成"""
    if llm is None:
        llm = get_llm()

    # 检索相关文档
    print("\n[RAG 步骤 1] 检索相关文档...")
    context_docs = retrieve(store, question)

    # 拼成上下文
    context = "\n\n".join([f"参考资料 {i+1}：{doc}" for i, doc in enumerate(context_docs)])

    # 构建 Prompt
    prompt = f"""请根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请回答"我不知道"。
回答要简洁准确，标注信息来源。

{context}

用户问题：{question}"""

    # 调用 LLM
    print("\n[RAG 步骤 2] 调用 LLM 生成回答...")
    answer = llm.chat([{"role": "user", "content": prompt}])

    return answer


# ============ 主入口 ============

def main():
    print("=" * 60)
    print("第七课：RAG 检索增强生成")
    print("RAG = Retrieval-Augmented Generation")
    print("=" * 60)

    store = build_vector_store()
    llm = get_llm()

    test_questions = [
        "RTX 5090 的价格是多少？",
        "DDR5 6000 和 DDR5 6400 有什么区别？",
        "3000 元预算买什么显卡好？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        answer = rag_answer(store, q, llm)
        print(f"\nRAG 回答：{answer}\n")


if __name__ == "__main__":
    main()
