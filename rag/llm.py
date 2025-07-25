import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPLATE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
       问题: {question}
       可参考的上下文：
       ···
       {context}
       ···
       如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，我不知道。
       有用的回答:""",
    InternLM_PROMPT_TEMPLATE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，我不知道。
        有用的回答:"""
)

class BaseModel:
    def __init__(self, path: str = ''):
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class GeminiChat(BaseModel):
    def __init__(self, path: str = '', model: str = 'gemini-2.5-flash'):
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], context: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("GEMINI_BASE_URL")
        )
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE']
                       .format(question=prompt, context=context)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=8192,
            temperature=0.7
        )
        return response.choices[0].message.content

class SiliconflowChat(BaseModel):
    def __init__(self, path: str = '', model: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], context: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL")
        )
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE']
                       .format(question=prompt, context=context)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=8192,
            temperature=0.7
        )
        return response.choices[0].message.content

