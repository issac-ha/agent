"""第七课运行入口：RAG 检索增强生成"""
import sys
import os

def _fix_syspath():
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path and os.path.isdir(sp):
            sys.path.insert(0, sp)

_fix_syspath()

from agent.lesson7_rag import build_vector_store, rag_answer, get_llm


def main():
    print("=" * 60)
    print("第七课：RAG 检索增强生成")
    print("输入问题提问，输入 /exit 退出")
    print("=" * 60)

    print("\n正在构建向量数据库...")
    store = build_vector_store()
    llm = get_llm()

    while True:
        question = input("\n请输入问题 > ").strip()
        if question == "/exit":
            print("再见")
            break
        if not question:
            continue

        answer = rag_answer(store, question, llm)
        print(f"\n回答：{answer}")


if __name__ == "__main__":
    main()
