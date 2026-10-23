from typing import List

class Chunker:
    def chunk_text(self, text: str, n: int, overlap: int) -> List[str]:
        """
        Chunks the given text into segments of n characters with overlap.

        Args:
        text (str): 文本
        n (int): 块长度
        overlap (int): 重叠度

        Returns:
        List[str]: A list of text chunks.
        """
        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i: i + n])
        return chunks