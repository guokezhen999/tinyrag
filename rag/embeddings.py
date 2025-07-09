import os
from typing import Optional, List

import numpy as np
from transformers import AutoModel, AutoTokenizer

class BaseEmbeddings:
    """ 嵌入模型的基类 """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

class BgeEmbedding(BaseEmbeddings):
    """ BGE嵌入模型 """
    def __init__(self, path: str = 'BAAI/bge-m3', is_api: bool = False, device: str = 'cpu') -> None:
        super().__init__(path, is_api)
        self.device = device
        self._model, self._tokenizer = self.load_model(path)

    def get_embedding(self, text: str) -> List[float]:
        import torch
        encoded_input = self._tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self._model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()

    def load_model(self, path: str):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(self.device)
        model.eval()
        return model, tokenizer

class BgeWithAPIEmbedding(BaseEmbeddings):
    """ 使用硅基流动API的嵌入模型 """
    def __init__(self, path: str = 'BAAI/bge-m3', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("SILICONFLOW_API_KEY")
            self.client.base_url = os.getenv("SILICONFLOW_BASE_URL")

    def get_embedding(self, text: str) -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=self.path).data[0].embedding
        else:
            raise NotImplementedError