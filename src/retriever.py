from pathlib import Path
from typing import List, Optional

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from mathlib_corpus import MathLibCorpus

_DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "mathlib_index"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class MathLibRetriever:
    """
    Hybrid FAISS + BM25 retriever with CrossEncoder reranking over Mathlib lemmas.

    On first use, call build() to create and persist the index.
    Subsequent runs load from disk automatically.
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.index_dir = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self._retriever = None
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

        if self._retriever is None:
            if not self.is_index_built():
                if not self._missing_index_warned:
                    print(
                        f"  [retriever] No FAISS index at {self.index_dir} — "
                        "skipping Mathlib RAG. The LLM will solve from its training "
                        "knowledge of Mathlib only. Run `python scripts/build_index.py` "
                        "to enable retrieval-augmented generation."
                    )
                    self._missing_index_warned = True
                return []
            self._load()
        results = self._retriever.invoke(query)
        return results[: k or self.rerank_top_k]

    def is_index_built(self) -> bool:
        return (self.index_dir / "index.faiss").exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=_EMBED_MODEL)

    def _load(self) -> None:
        if not self.is_index_built():
            raise RuntimeError(
                f"No FAISS index found at {self.index_dir}. "
                "Run `python scripts/build_index.py` first."
            )
        print("Loading FAISS index from disk…")
        embeddings = self._embeddings()
        faiss_store = FAISS.load_local(
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Re-build BM25 from FAISS docstore
        docs = list(faiss_store.docstore._dict.values())
        self._retriever = self._build_retriever(faiss_store, docs)

    def _build_retriever(self, faiss_store: FAISS, docs: List[Document]):
        faiss_retriever = faiss_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.top_k

        ensemble = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )

        cross_encoder = HuggingFaceCrossEncoder(model_name=_RERANK_MODEL)
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=self.rerank_top_k)

        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble,
        )
