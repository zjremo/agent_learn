import ast
import re

from openai import OpenAI
from typing import Dict, List
from .prompt import PLANNER_PROMPT, SOLVER_PROMPT
"""
models: 
1. LLMClient: 对话模型
2. Planner: 计划器
3. Solver: 解决器
"""
class LLMClient:
    def __init__(self, model_name: str, api_key: str, base_url: str, timeout: int = 10):
        self.model_name = model_name
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, message: List[Dict[str, str]], temperature: float = 0.5) -> str:
        print(f"{self.model_name} is thinking...")
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model_name, 
                messages=message, 
                temperature=temperature, 
                stream=True
            )

            print(f"{self.model_name} is thinking... done")

            chunks = []
            for chunk in resp:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                chunks.append(content)
            print()
            return "".join(chunks)
        except Exception as e:
            print(f"Error in {self.model_name} thinking: {e}")
            return None

class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT.format(question=question)
        print("Planner is planning...")
        response = self.llm_client.think([{"role": "user", "content": prompt}]) or ""
        print(f"Planner is planning... done, response: {response}")

        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            plan = match.group(1).strip()
            return ast.literal_eval(plan)
        else:
            print(f"Error in Planner planning: no plan found in response")
            return None

class Solver:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def solve(self, question: str, plan: List[str]) -> str:
        history = ""
        final_answer = ""

        # 起始编号从1开始
        for i, step in enumerate(plan, 1):
            print(f"Solver is solving... step {i}...")
            prompt = SOLVER_PROMPT.format(
                question=question,
                plan=plan,
                history=history if history else "None",
                current_step=step,
            )
            response = self.llm_client.think([{"role": "user", "content": prompt}]) or ""
            print(f"Solver is solving... step {i} done, response: {response}")
            history = history + f"Step {i}: {step}\nResponse: {response}\n"
            final_answer = response
            print(f"Solver is solving... step {i} done, final answer: {final_answer}")
        return final_answer
        