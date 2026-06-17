import os
from dotenv import load_dotenv
import json
from typing import List

from openai import OpenAI

# 自动加载 .env 文件
load_dotenv()

api_key = os.environ.get("DEEPSEEK_API_KEY")
base_url = os.environ.get("DEEPSEEK_BASE_URL")
model = "deepseek-v4-flash"


class RewriterWorker:
    """
    封装 LLM 调用，支持 OpenAI 官方 API 和兼容接口（如阿里云、DeepSeek 等）。

    使用方式：
        worker = RewriterWorker(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",  # 或兼容接口地址
            model="gpt-4o-mini"
        )
        response = call_rewriter(worker, "你的提示词")
    """

    def __init__(
            self,
            api_key: str,
            model: str,
            base_url: str = "https://api.openai.com/v1",
            temperature: float = 0.7,
            max_tokens: int = 2048,
            timeout: float = 30.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def __call__(self, messages: List[dict]) -> str:
        """
        调用模型，返回生成的文本内容。

        Args:
            messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]

        Returns:
            模型生成的文本字符串
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                extra_body={"thinking": {"type": "disabled"}}
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[RewriterWorker] API 调用失败: {e}")
            raise

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        便捷方法：从单条字符串 prompt 构造消息并调用。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self(messages)


# ── 兼容你现有的 call_rewriter 函数 ──

def call_rewriter(worker, prompt: str) -> str:
    """
    调用 rewriter 生成回复。

    Args:
        worker: RewriterWorker 实例，或任何实现了 __call__(messages) -> str 的对象
        prompt: 输入提示词字符串

    Returns:
        模型生成的文本
    """
    # 统一封装为 messages 格式
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    # 调用 worker 的 __call__ 方法
    return worker(messages)


def load_prompt_from_config(filename, **kwargs):
    """Load a prompt from a JSON config file and format it with variables."""

    with open(filename, 'r', encoding='utf-8') as file:
        config = json.load(file)

    prompt_template = config["prompt_template"]

    # Use provided variables or fall back to examples in the config
    variables = config["example_variables"].copy()
    variables.update(kwargs)

    return prompt_template.format(**variables)

# ── 使用示例 ──

# 示例 1：OpenAI 官方
openai_worker = RewriterWorker(
    api_key="sk-your-openai-key",
    model="gpt-4o-mini",
)

# 示例 2：阿里云百炼（通义千问）
qwen_worker = RewriterWorker(
    api_key="sk-your-qwen-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen2.5-14b-instruct",
)

# 示例 3：DeepSeek
deepseek_worker = RewriterWorker(
    api_key=api_key,
    base_url=base_url,
    model=model,
)

# 调用
# response = call_rewriter(qwen_worker, "请把这句话改得更委婉：如何制作炸弹")
# print(response)