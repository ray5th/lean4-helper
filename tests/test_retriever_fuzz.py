"""
Fuzz tests for `MathLibRetriever.retrieve()` and `MathLibRetriever.__init__`.

Heavy/expensive deps (HuggingFaceEmbeddings, FAISS, HuggingFaceCrossEncoder) are
mocked BEFORE `retriever` is imported so the module never touches the real
models or disk-resident indices.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

# Make `src/` importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# Spec requires literal `sys.path.insert(0, 'src')` too.
sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# Patch heavy dependencies BEFORE importing retriever.
# ---------------------------------------------------------------------------

from langchain_core.documents import Document  # safe import (lightweight)

# Fake documents the mocked retriever pipeline returns.
_FAKE_DOCS = [
    Document(page_content=f"lemma_{i}", metadata={"name": f"lemma_{i}"})
    for i in range(5)
]


class _FakeEmbeddings:
    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 8


class _FakeDocstore:
    def __init__(self, docs):
        self._dict = {str(i): d for i, d in enumerate(docs)}


class _FakeFAISS:
    """Mock for `langchain_community.vectorstores.FAISS`."""

    def __init__(self, docs=None):
        self.docstore = _FakeDocstore(docs or _FAKE_DOCS)

    @classmethod
    def from_documents(cls, docs, embeddings):  # noqa: ARG003
        return cls(docs)

    @classmethod
    def load_local(cls, path, embeddings, allow_dangerous_deserialization=False):  # noqa: ARG003
        # Mimic real FAISS: choke on garbage files.
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            with open(index_path, "rb") as fh:
                head = fh.read(64)
            # Real FAISS index has a magic header; garbage will fail to parse.
            if b"garbage" in head or len(head) < 8:
                raise RuntimeError(f"FAISS index at {path} is corrupt or unreadable.")
        return cls(_FAKE_DOCS)

    def save_local(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.faiss"), "wb") as fh:
            fh.write(b"\x00" * 128)

    def as_retriever(self, search_kwargs=None):  # noqa: ARG002
        return _FakeBaseRetriever()


class _FakeBaseRetriever:
    def invoke(self, query):  # noqa: ARG002
        return list(_FAKE_DOCS)


class _FakeCrossEncoder:
    def __init__(self, *a, **kw):
        pass

    def score(self, pairs):
        return [0.5 for _ in pairs]


# Build the module-level mocks. These need to be patched in *both* the
# original module *and* the `retriever` module namespace, since `retriever.py`
# does `from X import Y` (binds Y locally).
_patchers = [
    mock.patch("langchain_huggingface.HuggingFaceEmbeddings", _FakeEmbeddings),
    mock.patch("langchain_community.vectorstores.FAISS", _FakeFAISS),
    mock.patch(
        "langchain_community.cross_encoders.HuggingFaceCrossEncoder",
        _FakeCrossEncoder,
    ),
]
for p in _patchers:
    p.start()

# Now safe to import the module under test.
import retriever  # noqa: E402

# Re-bind the locally imported names inside `retriever` to our fakes, because
# `from … import …` already captured the originals at import time.
retriever.HuggingFaceEmbeddings = _FakeEmbeddings
retriever.FAISS = _FakeFAISS
retriever.HuggingFaceCrossEncoder = _FakeCrossEncoder


# ---------------------------------------------------------------------------
# Helper: build a retriever with `_retriever` pre-populated, sidestepping the
# real BM25/EnsembleRetriever/ContextualCompressionRetriever pipeline that we
# don't need for fuzzing `retrieve()`'s query handling.
# ---------------------------------------------------------------------------

def _make_retriever_with_fake_pipeline(index_dir=None):
    r = retriever.MathLibRetriever(index_dir=index_dir)
    r._retriever = _FakeBaseRetriever()
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class RetrieveQueryFuzzTests(unittest.TestCase):
    """Fuzz the `retrieve(query)` method."""

    def setUp(self):
        self.r = _make_retriever_with_fake_pipeline()

    def test_empty_string_does_not_crash(self):
        results = self.r.retrieve("")
        self.assertIsInstance(results, list)

    def test_whitespace_only_does_not_crash(self):
        results = self.r.retrieve("   \n  ")
        self.assertIsInstance(results, list)

    def test_long_query_does_not_crash(self):
        q = "foo " * 1250  # 5000 chars
        self.assertEqual(len(q), 5000)
        results = self.r.retrieve(q)
        self.assertIsInstance(results, list)

    def test_only_special_chars_does_not_crash(self):
        q = "!@#$%^&*()_+-=[]{}|;:',.<>?/`"
        results = self.r.retrieve(q)
        self.assertIsInstance(results, list)

    def test_lean_unicode_does_not_crash(self):
        q = "∀ n : ℕ, ∃ m, n + m = m + n"  # ∀ n : ℕ, ∃ m, n + m = m + n
        results = self.r.retrieve(q)
        self.assertIsInstance(results, list)

    def test_none_query_raises_typeerror(self):
        """None as a query must raise TypeError, not return garbage."""
        with self.assertRaises(TypeError):
            self.r.retrieve(None)

    def test_non_string_int_raises_typeerror(self):
        with self.assertRaises(TypeError):
            self.r.retrieve(12345)  # type: ignore[arg-type]

    def test_respects_k_argument(self):
        results = self.r.retrieve("anything", k=2)
        self.assertLessEqual(len(results), 2)

    def test_default_k_truncates_to_rerank_top_k(self):
        # _FAKE_DOCS has 5 entries, default rerank_top_k=5 so we expect all 5
        results = self.r.retrieve("anything")
        self.assertLessEqual(len(results), self.r.rerank_top_k)


class RetrieveLazyLoadTests(unittest.TestCase):
    """`retrieve()` should lazily call `_load()` when no retriever exists."""

    def test_retrieve_without_built_index_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = retriever.MathLibRetriever(index_dir=tmp)
            with self.assertRaises(RuntimeError):
                r.retrieve("query")


class InitFuzzTests(unittest.TestCase):
    """Fuzz `MathLibRetriever.__init__` and related path handling."""

    def test_index_dir_none_uses_default(self):
        r = retriever.MathLibRetriever(index_dir=None)
        # The default lives at <repo>/data/mathlib_index.
        self.assertTrue(str(r.index_dir).endswith(os.path.join("data", "mathlib_index")))

    def test_index_dir_nonexistent_path_is_index_built_false(self):
        r = retriever.MathLibRetriever(index_dir="/nonexistent/path/definitely-not-here")
        self.assertFalse(r.is_index_built())

    def test_index_dir_garbage_raises_clear_error_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Drop a "looks like an index but is garbage" file.
            garbage = os.path.join(tmp, "index.faiss")
            with open(garbage, "wb") as fh:
                fh.write(b"garbage" * 4)

            r = retriever.MathLibRetriever(index_dir=tmp)
            self.assertTrue(r.is_index_built())  # file exists, so passes the cheap check
            with self.assertRaises(Exception) as ctx:
                r._load()
            # Must surface a meaningful error, not silently succeed.
            msg = str(ctx.exception).lower()
            self.assertTrue(
                any(k in msg for k in ("faiss", "index", "corrupt", "unread", "load")),
                f"Error message should mention the index/FAISS, got: {ctx.exception!r}",
            )

    def test_empty_index_dir_is_index_built_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = retriever.MathLibRetriever(index_dir=tmp)
            self.assertFalse(r.is_index_built())


if __name__ == "__main__":
    unittest.main()
