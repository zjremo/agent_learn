import logging
import os
from .consts import EnvKeys, logging_level_map
from openai import OpenAI
from typing import List, Dict

def setLogger():
    # getenv 未设置时是 None；str(None)=="none"，不能用 or 兜底
    level = (os.getenv(EnvKeys.LOGGING_LEVEL) or "debug").lower()
    logging_level = logging_level_map.get(level, logging.DEBUG)
    logger.setLevel(logging_level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger = logging.getLogger(__name__)
setLogger()

class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, timeout = 10):
        api_key = api_key or os.getenv(EnvKeys.LLM_API_KEY)
        base_url = base_url or os.getenv(EnvKeys.LLM_BASE_URL)

        self.model_name = model_name or os.getenv(EnvKeys.LLM_MODEL)    
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        logger.info(f"{self.model_name} is thinking...")

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            logger.info(f"{self.model_name} think done...")

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