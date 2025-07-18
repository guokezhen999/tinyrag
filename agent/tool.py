import os

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

from rag.vector_base import VectorStore
from rag.embeddings import BgeEmbedding
from rag.llm import PROMPT_TEMPLATE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_deep_learning_knowledge(query: str) -> str:
    """ 查询深度学习相关知识，生成提示词。 """
    vector = VectorStore()
    vector.load_vector(file='d2l')
    embedding = BgeEmbedding(device='mps')
    content = vector.query(query, EmbeddingModel=embedding, k=1)[0]
    prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE'].format(question=query, context=content)
    return prompt

d2l_tool = Tool(
    name="deep_learning_knowledge",
    func=get_deep_learning_knowledge,
    description="这个 RAG 系统用于查询深度学习的知识。它会尝试在内部知识库中查找，并找到最相关的段落生成提示词。"
                "当需要查询深度学习相关知识时候，请调用这个函数。"
)

search = GoogleSearchAPIWrapper()

search_tool = Tool(
    name="google_search",
    func=search.run,
    description="当需要联网查询时，请使用此工具进行搜索。"
)