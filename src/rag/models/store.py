import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Metadata:
    type: Optional[str] = None
    chunk_index: int = -1
    original_chunk: Optional[str] = None

class SimpleVectorDB:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata: Metadata=None):
        self.vectors.append(embedding)
        self.texts.append(text)
        self.metadata.append(metadata or Metadata())

    def similarity_search(self, query_embedding, k=5) -> List[Dict[str, Any]]:
        """
        查找与查询嵌入向量最相似的项目。

        Args:
        query_embedding (List[float]): 查询嵌入向量。
        k (int): 返回的结果数量。

        Returns:
        List[Dict]: 最相似的前 k 个项目及其文本和元数据。
        """
        if not self.vectors:
            return []

        query_vector = np.array(query_embedding)
        similarities = []
        for i, vector in enumerate(self.vectors):  # 遍历所有向量
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 存储索引和相似度

        # 排序搜索最相似的Top k
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回Top k 个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results