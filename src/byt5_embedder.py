"""
ByT5-based query embedder for the LeanDojo Lean 4 premise retriever.

Wraps `kaiyuy/leandojo-lean4-retriever-byt5-small` so it conforms to LangChain's
Embeddings interface and can drop into the existing FAISS retriever pipeline.

The encoder was trained on (proof state, used premise) pairs from Mathlib, so
its representations are tuned for premise selection — significantly stronger
than general-English encoders like `all-MiniLM-L6-v2` for this domain.
"""
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

_MODEL_NAME = "kaiyuy/leandojo-lean4-retriever-byt5-small"


class ByT5PremiseEmbedder(Embeddings):
    """
    Lean-aware query embedder. Loads the LeanDojo ByT5 encoder once on
    construction and L2-normalizes outputs for cosine-similarity search.
    """

    def __init__(self, device: str = "cpu", max_length: int = 1024):
        # Lazy import so unit tests that mock this class don't need transformers
        # at import time.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModel.from_pretrained(_MODEL_NAME).to(device)
        self._model.eval()
        self._device = device
        self._max_length = max_length

    def _encode(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            out = self._model.encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()
