from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from byt5_embedder import ByT5PremiseEmbedder
from mathlib_corpus import MathLibCorpus

_DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "mathlib_index"
# Query-time embedder. Lean-aware ByT5 encoder from LeanDojo, trained on
# (proof_state, used_premise) pairs from Mathlib. The Mathlib FAISS index is
# built from LeanDojo's pre-computed embeddings via
# `scripts/build_leandojo_index.py`.
#
# History: we previously hybrid-ensembled this with BM25 and reranked with a
# generic `ms-marco-MiniLM-L-6-v2` cross-encoder. That made sense for the old
# general-English `all-MiniLM-L6-v2` dense path. With the LeanDojo encoder the
# semantic results are already strong and domain-tuned — the generic reranker
# was actively reordering correct premises (Nat.add_comm, Nat.add_assoc, …)
# below irrelevant matches (Ackermann, ZMod, choose). We dropped both layers.


class MathLibRetriever:
    """
    Lean-aware FAISS retriever over Mathlib premises using LeanDojo's ByT5
    encoder. Loads an IVFPQ-compressed FAISS index built from LeanDojo's
    pre-computed premise embeddings (see scripts/build_leandojo_index.py).
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        top_k: int = 20,
        rerank_top_k: int = 5,
        nprobe: int = 32,
    ):
        self.index_dir = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
        self.top_k = top_k
        # `rerank_top_k` kept for back-compat; it just caps the number returned
        # from the underlying FAISS search (we no longer rerank).
        self.rerank_top_k = rerank_top_k
        self.nprobe = nprobe
        self._retriever = None
        self._faiss_store: Optional[FAISS] = None
        self._missing_index_warned = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, mathlib_root: Optional[str] = None, max_files: Optional[int] = None) -> None:
        """
        Extract Mathlib documents, build FAISS + BM25 indices, and persist to disk.
        Call this once (via scripts/build_index.py) before first use.
        """
        print("Extracting Mathlib corpus…")
        corpus = MathLibCorpus(mathlib_root=mathlib_root)
        docs = corpus.extract(max_files=max_files)
        print(f"  {len(docs)} declarations extracted.")

        embeddings = self._embeddings()

        print("Building FAISS index…")
        faiss_store = FAISS.from_documents(docs, embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss_store.save_local(str(self.index_dir))
        print(f"  Index saved to {self.index_dir}")

        self._retriever = self._build_retriever(faiss_store, docs)

    # Hard cap on query length passed to downstream retrievers. Long queries
    # waste embedding work and can blow past tokenizer limits; truncating here
    # gives callers a predictable upper bound.
    _MAX_QUERY_CHARS = 2000

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve and rerank the most relevant Mathlib lemmas for a query.

        Args:
            query: Natural-language or Lean-syntax query (e.g., proof goals + errors).
            k: Number of results to return after reranking (defaults to self.rerank_top_k).

        Returns:
            List of Documents ranked by relevance.

        Raises:
            TypeError: If `query` is not a string.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a str, got {type(query).__name__}"
            )

        # Empty/whitespace-only queries are valid input but degenerate; bail
        # out early with no results rather than asking the embedding model to
        # vectorise an empty string (which produces noisy nearest neighbours).
        if not query.strip():
            return []

        # Truncate absurdly long inputs so a runaway caller can't pin the
        # embedding model.
        if len(query) > self._MAX_QUERY_CHARS:
            query = query[: self._MAX_QUERY_CHARS]

        if self._faiss_store is None:
            if not self.is_index_built():
                if not self._missing_index_warned:
                    print(
                        f"  [retriever] No FAISS index at {self.index_dir} — "
                        "skipping Mathlib RAG. The LLM will solve from its training "
                        "knowledge of Mathlib only. Run `python scripts/build_leandojo_index.py` "
                        "to enable retrieval-augmented generation."
                    )
                    self._missing_index_warned = True
                return []
            self._load()
        return self._faiss_store.similarity_search(query, k=k or self.rerank_top_k)

    def is_index_built(self) -> bool:
        return (self.index_dir / "index.faiss").exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embeddings(self) -> ByT5PremiseEmbedder:
        return ByT5PremiseEmbedder()

    def _load(self) -> None:
        if not self.is_index_built():
            raise RuntimeError(
                f"No FAISS index found at {self.index_dir}. "
                "Run `python scripts/build_leandojo_index.py` first."
            )
        print("Loading FAISS index from disk…")
        embeddings = self._embeddings()
        self._faiss_store = FAISS.load_local(
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Tune IVFPQ search breadth. nprobe=32 / nlist=512 = 6% of clusters
        # searched — good recall/speed tradeoff for this index size.
        try:
            self._faiss_store.index.nprobe = self.nprobe
        except AttributeError:
            pass  # not an IVF index — nothing to tune
