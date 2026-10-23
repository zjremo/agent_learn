from dotenv import load_dotenv

# 必须在 import rag 之前加载，否则模块顶层的 setLogger/os.getenv 读不到变量
load_dotenv(dotenv_path=".ragenv")

from rag.models import SimpleRag

if __name__ == "__main__":
    query = "南开大学学业奖学金的奖励标准"
    simple_rag = SimpleRag()
    # 1. store document in db
    simple_rag.process_document(file_path="data/test.pdf")

    # 2. get LLM reply with rag
    resp = simple_rag.reply_by_rag(query=query, k=3)
    print(resp)