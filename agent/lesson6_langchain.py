"""
第六课：LangChain 框架入门

核心组件：
  1. Chat Models - 封装 LLM 调用
  2. Prompt Templates - 提示词模板
  3. Chains - 链式调用（LCEL）
  4. Memory - 记忆管理
  5. Agents + Tools - 智能体和工具

运行：python run_lesson6.py
"""
import sys
import os

# 修复 Windows 下 site-packages 不在 sys.path 的问题
def _fix_syspath():
    import site
    for sp in site.getsitepackages():
        if sp not in sys.path and os.path.isdir(sp):
            sys.path.insert(0, sp)

_fix_syspath()

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


# ============ 1. Chat Models：封装 LLM 调用 ============

def demo_chat_model():
    """用 LangChain 封装 LLM，比直接调 API 更简洁"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0.7,
    )

    messages = [
        SystemMessage(content="你是一个简洁的 AI 助手。"),
        HumanMessage(content="用一句话解释什么是 Agent"),
    ]

    response = llm.invoke(messages)
    print(f"LLM 回复: {response.content}\n")


# ============ 2. Prompt Templates：提示词模板 ============

def demo_prompt_template():
    """用模板管理提示词，避免硬编码"""
    # 模板：用 {变量} 占位
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请用{style}的风格回答。"),
        ("human", "{question}"),
    ])

    # 填充模板
    formatted = prompt.format_messages(
        role="数学老师",
        style="幽默",
        question="什么是质数？",
    )
    print("填充后的消息：")
    for m in formatted:
        print(f"  [{m.type}] {m.content}")
    print()


# ============ 3. Chains：链式调用（LCEL）============

def demo_chain():
    """
    LCEL = LangChain Expression Language
    用 | 符号把组件串起来：prompt | llm | parser
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手，把中文翻译成英文。"),
        ("human", "{text}"),
    ])

    # 链式调用：模板 -> 模型 -> 解析器
    chain = prompt | llm | StrOutputParser()

    # 调用链，只需要传变量
    result = chain.invoke({"text": "人工智能正在改变世界"})
    print(f"翻译结果: {result}\n")


# ============ 4. Memory：记忆管理 ============

def demo_memory():
    """LangChain 的消息历史管理"""
    # InMemoryChatMessageHistory 替代了旧版的 ConversationBufferMemory
    history = InMemoryChatMessageHistory()

    # 模拟多轮对话
    history.add_user_message("我叫小明")
    history.add_ai_message("你好小明")
    history.add_user_message("我喜欢编程")
    history.add_ai_message("很棒")

    # 查看记忆
    print("记忆内容：")
    for msg in history.messages:
        print(f"  [{msg.type}] {msg.content}")
    print()


# ============ 5. Agents + Tools：智能体和工具 ============

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '100+200*2'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"错误: {e}"


@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    db = {"北京": "晴 22°C", "上海": "多云 25°C", "广州": "小雨 28°C"}
    return db.get(city, f"暂无{city}的天气数据")


def demo_agent():
    """用 LangChain 创建带工具的 Agent"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    tools = [calculator, get_weather]

    # LangChain 1.x 用 create_agent，参数更简洁
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个有用的助手，可以使用工具。",
    )

    # 运行 Agent，输入是消息列表
    result = agent.invoke({"messages": [HumanMessage(content="北京和上海的天气分别是什么？")]})

    # 取最后一条消息作为回答
    final_answer = result["messages"][-1].content
    print(f"\n最终回答: {final_answer}\n")


# ============ 主入口 ============

def main():
    print("=" * 60)
    print("第六课：LangChain 框架入门")
    print("核心组件：ChatModel / PromptTemplate / Chain / Memory / Agent")
    print("=" * 60)

    demos = [
        ("1. Chat Models（LLM 封装）", demo_chat_model),
        ("2. Prompt Templates（提示词模板）", demo_prompt_template),
        ("3. Chains（链式调用 LCEL）", demo_chain),
        ("4. Memory（记忆管理）", demo_memory),
        ("5. Agents + Tools（智能体）", demo_agent),
    ]

    for name, func in demos:
        print(f"\n{'='*60}")
        print(name)
        print(f"{'='*60}")
        func()


if __name__ == "__main__":
    main()
