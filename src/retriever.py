"""
retriever.py

Builds and queries a FAISS index over Mathlib4 lemma docstrings.

Usage:
    # Build index from a jsonl file of lemmas:
    build_index("data/mathlib_lemmas.jsonl", "data/mathlib.index")

    # Query at runtime:
    retriever = Retriever("data/mathlib.index", "data/mathlib_lemmas.jsonl")
    lemmas = retriever.retrieve("commutativity of addition", k=5)
"""

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "data/mathlib.index"
META_FILE = "data/mathlib_meta.pkl"


class Retriever:
    """
    Manages the FAISS index and performs semantic retrieval of Mathlib4 lemmas.
    """

    def __init__(
        self,
        index_path: str = INDEX_FILE,
        meta_path: str = META_FILE,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)  # list of dicts: {name, type, doc}

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Return the top-k most relevant Mathlib4 lemmas for a given query.

        Args:
            query: Natural language or Lean theorem statement.
            k:     Number of results to return.

        Returns:
            List of lemma dicts with keys 'name', 'type', 'doc'.
        """
        embedding = self.model.encode([query], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)
        _, indices = self.index.search(embedding, k)
        return [self.metadata[i] for i in indices[0] if i < len(self.metadata)]


def build_index(
    lemmas_jsonl: str = "data/mathlib_lemmas.jsonl",
    index_path: str = INDEX_FILE,
    meta_path: str = META_FILE,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """
    Build a FAISS index from a JSONL file of Mathlib4 lemmas.

    Each line in `lemmas_jsonl` should be a JSON object with:
        - "name": lemma name (str)
        - "type": Lean type signature (str)
        - "doc":  docstring or description (str, optional)

    Args:
        lemmas_jsonl: Path to the input JSONL file.
        index_path:   Where to save the FAISS index file.
        meta_path:    Where to save the metadata pickle file.
        model_name:   SentenceTransformer model to use for embeddings.
    """
    print(f"Loading lemmas from {lemmas_jsonl}...")
    metadata = []
    texts = []
    with open(lemmas_jsonl, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            metadata.append(entry)
            # Build a rich text representation for embedding
            text = f"{entry.get('name', '')} : {entry.get('type', '')}. {entry.get('doc', '')}"
            texts.append(text)

    print(f"Loaded {len(texts)} lemmas. Generating embeddings...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (normalized)
    index.add(embeddings)

    os.makedirs(Path(index_path).parent, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index saved to {index_path} ({len(texts)} vectors, dim={dim})")


if __name__ == "__main__":
    build_index()
