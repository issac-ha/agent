"""
真实大模型 API 客户端
==================================================
封装 OpenAI 兼容格式的 API 调用，支持：
  - 从 .env 文件读取配置
  - 对话补全（chat completion）
  - 流式输出（streaming）
  - 结构化输出（JSON mode）

依赖：pip install openai python-dotenv
"""
import os
from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv()


class LLMClient:
    """大模型 API 客户端（OpenAI 兼容格式）"""

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("请在 .env 文件中配置 LLM_API_KEY")

        # 延迟导入，避免没装 openai 时整个模块报错
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """对话补全：传入消息列表，返回模型回复文本

        Args:
            messages: 消息列表，格式：
                [
                    {"role": "system", "content": "你是..."},
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    {"role": "user", "content": "..."}
                ]
            temperature: 随机性，0=确定性，1=最随机
            max_tokens: 最大生成 token 数

        Returns:
            模型回复的文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def chat_stream(self, messages: list, temperature: float = 0.7):
        """流式输出：逐字返回模型回复（适合打字机效果）

        Yields:
            每次返回一段文本片段
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_json(self, messages: list) -> dict:
        """结构化输出：让模型返回 JSON（需要模型支持）

        Returns:
            解析后的字典
        """
        import json
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


# 全局单例（方便其他模块直接 import）
_llm = None


def get_llm() -> LLMClient:
    """获取 LLM 客户端单例"""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
