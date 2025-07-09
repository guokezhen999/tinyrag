import json
from typing import List

from rag.embeddings import BaseEmbeddings, BgeWithAPIEmbedding

import os
import numpy as np
from tqdm import tqdm

class VectorStore:
    def __init__(self, document: List[str] = ['']):
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage', file: str = 'file'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f"{file}_doc.json"), 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(os.path.join(path, f"{file}_vec.json"), 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage', file: str = 'file'):
        with open(os.path.join(path, f"{file}_doc.json"), 'r', encoding='utf-8') as f:
            self.document = json.load(f)
        with open(os.path.join(path, f"{file}_vec.json"), 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector) for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
