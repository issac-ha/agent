"""
大模型 API 客户端（OpenAI 兼容格式）

支持：对话补全、流式输出、结构化输出（JSON）
依赖：pip install openai python-dotenv
"""
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("请在 .env 文件中配置 LLM_API_KEY")

        # 延迟导入，避免没装 openai 时报错
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages, temperature=0.7, max_tokens=2000):
        """
        对话补全

        Args:
            messages: 消息列表 [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 随机性，0=确定，1=随机
            max_tokens: 最大生成 token 数
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def chat_stream(self, messages, temperature=0.7):
        """流式输出，逐段返回文本"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_json(self, messages):
        """让模型返回 JSON，解析后返回字典"""
        import json
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


# 全局单例
_llm = None


def get_llm():
    """获取 LLM 客户端单例"""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
