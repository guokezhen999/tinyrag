from typing import List
import numpy as np

class BaseRanker:
    def __init__(self, path: str):
        self.path = path

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        raise NotImplemented

class BgeReranker(BaseRanker):
    def __init__(self, path: str = 'BAAI/bge-reranker-base', device: str = 'cpu'):
        self.device = device
        self._model, self._tokenizer = self.load_model(path)

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        import torch
        pairs = [(text, c) for c in content]
        with torch.no_grad():
            inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
            index = np.argsort(scores.tolist())[-k:][::-1]
        return [content[i] for i in index]

    def load_model(self, path: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        model.eval()
        return model, tokenizer