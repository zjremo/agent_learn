from typing import List, Dict, Any, Optional
from openai import OpenAI

from .prompts import REFLECT_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE

"""
Models 模型层

- Memory 记忆模块
- LLMClient 大模型客户端
- Reflection 反思模块
- Refiner 优化模块
"""

# Memory 记忆模块
class Memeory:
    def __init__(self):
        self.records : List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        添加一条记忆
        
        args:
        record_type: 记忆类型
        content: 记忆内容
        """
        record = {
            "type": record_type,
            "content": content
        }
        self.records.append(record)
        print(f'update memory, type: {record_type}, content: {content}')

    def get_trajectory(self) -> str:
        """
        所有记忆记录格式化

        return:
        格式化记忆字符串
        """
        trajectory_parts = []
        for record in self.records:
            record_type = record["type"]
            match record_type :
                case 'execution':
                    trajectory_parts.append(f'---上一轮尝试代码---\n{record["content"]}')
                case 'reflection':
                    trajectory_parts.append(f'---评审员反馈---\n{record["content"]}')
        return "/n/n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最后一条执行记录

        return:
        None or a record of execution
        """
        for record in reversed(self.records):
            if record["type"] == 'execution':
                return record["content"]
        return None

# LLM客户端 
class LLMClient:
    def __init__(self, api_key, base_url, model_name, timeout:int = 10):
        self.model_name = model_name
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        print(f"{self.model_name} is thinking...")

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            print(f"{self.model_name} think done...")

            chunks = []
            for chunk in resp:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end='', flush=True)
                chunks.append(content)
            print()
            return "".join(chunks)
        except Exception as e:
            print(f"Error in {self.model_name} thinking: {e}")
            return None

class Reflection:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    """
    反思，并给出下一步优化建议
    task : 原始任务描述 
    execution: 上一轮的执行代码
    """
    def reflect(self, task: str, execution: str) -> str:
        user_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=execution)

        messages = [
            {"role": "user", "content": user_prompt}
        ]
        return self.llm_client.think(messages=messages)

class Refiner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    """
    优化代码

    args: 
    task 原始任务描述
    last_execution 上一轮尝试代码
    feedback 评估器反馈
    """
    def refine(self, task: str, last_execution: str, feedback: str):
        user_prompt = REFINE_PROMPT_TEMPLATE.format(task=task, last_code_attempt=last_execution, feedback=feedback)

        messages = [
            {"role": "user", "content": user_prompt}
        ]
        return self.llm_client.think(messages)