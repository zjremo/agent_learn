import logging
import os
from tqdm import tqdm
from typing import List, Dict, Any

from .chunker import Chunker
from .consts import logging_level_map, EnvKeys 
from .chat_llm import LLMClient
from .embedding_llm import EmbeddingClient
from .pdf_reader import PdfReader
from .prompts import RAG_SYSTEM_PROMPT
from .store import SimpleVectorDB, Metadata


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

class SimpleRag:
    def __init__(self):
        self.chunker = Chunker()
        self.llm_client = LLMClient()
        self.embedding_client = EmbeddingClient()
        self.pr = PdfReader()
        self.db = SimpleVectorDB()

    def process_document(self, file_path: str, chunk_size = 1000, chunk_overlap = 200):
        """
        process document and build storage

        Args:
        file_path (str): the path of pdf document 

        Returns:
        None
        """
        logger.info("SimpleRag process_document...")
        # 1. get all_text by pdf_reader
        all_text = self.pr.extract_text_from_pdf(pdf_path=file_path)
        logger.info(f"SimpleRag extract text from pdf, all_text is: {all_text}")

        # 2. get chunks
        chunk_size = int(os.getenv(EnvKeys.CHUNK_SIZE) or chunk_size)
        chunk_overlap = int(os.getenv(EnvKeys.CHUNK_OVERLAP) or chunk_overlap)
        text_chunks = self.chunker.chunk_text(all_text, chunk_size, chunk_overlap)

        # 3. store to vectordb
        for i, chunk in enumerate(tqdm(text_chunks, desc="处理文本块")):
            chunk_embedding_resp = self.embedding_client.create_embeddings(chunk)
            chunk_embedding = chunk_embedding_resp.data[0].embedding

            self.db.add_item(
                text=chunk,
                embedding=chunk_embedding,
                metadata=Metadata(type="chunk", chunk_index=i)
            )
        logger.info("SimpleRag process_document done...")
        
    def semantic_search(self, query: str, k=5) -> List[Dict[str, Any]]:
        logger.info("SimpleRag semantic_search...")
        query_embedding_resp = self.embedding_client.create_embeddings(query)
        query_embedding = query_embedding_resp.data[0].embedding

        results = self.db.similarity_search(query_embedding=query_embedding, k=k)
        logger.info(f"SimpleRag get similar chunks, results: {results}")
        logger.info("SimpleRag semantic_search done...")
        return [result["text"] for result in results]

    def generate_response(self, query: str, top_chunks: List[str], system_prompt: str = RAG_SYSTEM_PROMPT) -> str:
        logger.info("SimpleRag generate_response...")
        user_prompt = "\n".join([f"context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
        user_prompt = f"{user_prompt}\nquestion:{query}"

        logger.info(f"After Given context, SimpleRag user_prompt:\n{user_prompt}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        resp = self.llm_client.think(messages=messages, temperature=0.5)
        logger.info("SimpleRag generate_response done...")
        return resp

    def reply_by_rag(self, query: str, k: int=5) -> str:
        logger.info("SimpleRag reply_by_rag...")
        top_chunks = self.semantic_search(query=query, k=k) 
        logger.info("SimpleRag reply_by_rag done...")
        return self.generate_response(query=query, top_chunks=top_chunks)
    