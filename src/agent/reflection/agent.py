import os
from typing import Optional
from .models import LLMClient, Reflection, Refiner, Memeory

class ReflectionAgent:
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout:int = 10, max_iteration:int = 5):
        model_name = model_name if model_name else os.getenv("OPENAI_MODEL")
        api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")

        self.llm_client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            timeout=timeout
        )
        self.reflection = Reflection(self.llm_client)
        self.refiner = Refiner(self.llm_client)
        self.memory = Memeory()
        self.max_iteration = max_iteration

    def run(self, question:str):
        # 先执行第一步
        INITIAL_PROMPT_TEMPLATE = """
        你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
        你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

        要求: {task}

        请直接输出代码，不要包含任何额外的解释。
        """
        print("正在进行初始尝试...")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=question)
        initial_code = self.llm_client.think(messages=[{"role": "user", "content": initial_prompt}])
        self.memory.add_record(record_type="execution", content=initial_code)

        for i in range(self.max_iteration):
            print(f"--- 第{i + 1}轮迭代 ---")

            # a. 反思
            print("正在进行反思...")
            last_execution = self.memory.get_last_execution()
            feedback = self.reflection.reflect(task=question, execution=last_execution)
            self.memory.add_record(record_type="reflection", content=feedback)

            # b. 检查跳出条件
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("反思认为代码不需要再进行修改...")
                break;

            # c. 优化
            refined_code = self.refiner.refine(question, last_execution=last_execution, feedback=feedback)
            self.memory.add_record(record_type="execution", content=refined_code)

        final_code = self.memory.get_last_execution()
        print(f"---任务完成---\n最终生成代码为: \n{final_code}")
        return final_code