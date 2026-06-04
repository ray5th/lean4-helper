#!/usr/bin/env python3
"""
Build a Lean-aware FAISS index from LeanDojo's pre-computed Mathlib premise
embeddings.

Downloads `kaiyuy/premise-embeddings-leandojo-lean4-retriever-byt5-small`
(180,973 premises × 1472-dim ByT5 embeddings, ~1 GB) and compresses it into
an IVFPQ FAISS index (~15 MB) wrapped in a LangChain vectorstore that uses
ByT5 at query time.

The IVFPQ compression keeps the committed index under GitHub's free LFS
quota with ~95% recall vs flat search.

Usage:
    python scripts/build_leandojo_index.py
"""
import json
import os
import sys
from pathlib import Path

# faiss and torch both bring their own libomp.dylib on macOS and collide on
# import. This env var is the documented workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

EMBEDDINGS_REPO = "kaiyuy/premise-embeddings-leandojo-lean4-retriever-byt5-small"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mathlib_index"


def main():
    print("Downloading LeanDojo premise embeddings (may be cached)…")
    emb_path = hf_hub_download(EMBEDDINGS_REPO, "embeddings.npy")
    dict_path = hf_hub_download(EMBEDDINGS_REPO, "dictionary.json")

    print("Loading embeddings…")
    embeddings = np.load(emb_path).astype(np.float32)
    print(f"  {embeddings.shape} ({embeddings.nbytes / 1e6:.0f} MB float32)")

    # L2-normalize so cosine == inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    print("Loading dictionary…")
    with open(dict_path) as f:
        dict_entries = json.load(f)
    print(f"  {len(dict_entries)} premises")

    print("Building langchain Documents…")
    docs = []
    for i in range(len(embeddings)):
        entry = dict_entries[str(i)]
        docs.append(
            Document(
                page_content=entry["code"],
                metadata={
                    "name": entry["full_name"],
                    "path": entry["path"],
                },
            )
        )

    # IVFPQ — product-quantized inverted-file index. ~95% recall, ~60-100x smaller.
    d = embeddings.shape[1]  # 1472
    nlist = 512  # number of coarse clusters (~sqrt(N))
    m = 64       # PQ segments (must divide d)
    nbits = 8    # bits per code
    print(f"Training IVFPQ index (d={d}, nlist={nlist}, m={m}, nbits={nbits})…")
    quantizer = faiss.IndexFlatIP(d)  # inner product after L2 normalize = cosine
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 16  # search 16 clusters at query time

    # Free the raw embeddings before loading ByT5 (cuts peak RAM by ~1GB).
    del embeddings

    print("Wrapping into LangChain FAISS vectorstore…")
    # Import here so torch is loaded *after* FAISS training is done.
    from byt5_embedder import ByT5PremiseEmbedder  # noqa: E402
    embedder = ByT5PremiseEmbedder()
    docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    vectorstore = LCFAISS(
        embedding_function=embedder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    print(f"Saving to {OUT_DIR}…")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(OUT_DIR))

    print("Output files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
