import os
from openai import OpenAI

from .consts import EnvKeys

class EmbeddingClient:
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, timeout = 10):
        api_key = api_key or os.getenv(EnvKeys.EMBEDDING_API_KEY)
        base_url = base_url or os.getenv(EnvKeys.EMBEDDING_BASE_URL)

        self.model_name = model_name or os.getenv(EnvKeys.EMBEDDING_MODEL)    
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def create_embeddings(self, text:str):
        resp = self.client.embeddings.create(
            model = self.model_name,
            input=text
        )
        return resp