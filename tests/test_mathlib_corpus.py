"""
Unit tests for `src/mathlib_corpus.py`.

Covers:
- `_parse_lean_file` regex behaviour across synthetic Lean snippets.
- `_find_mathlib_root` fallback behaviour when nothing matches.
- `MathLibCorpus.extract` happy path, error path, and `max_files` slicing.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make `src/` importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# Spec requires literal `sys.path.insert(0, 'src')` too.
sys.path.insert(0, "src")

from mathlib_corpus import (  # noqa: E402
    MathLibCorpus,
    _find_mathlib_root,
    _parse_lean_file,
)


def _write(tmpdir: str, name: str, content: str) -> str:
    """Write content to a tempfile and return its path."""
    p = Path(tmpdir) / name
    p.write_text(content, encoding="utf-8")
    return str(p)


class ParseLeanFileTest(unittest.TestCase):
    """Synthetic fuzz corpus for `_parse_lean_file`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_theorem(self):
        path = _write(self.tmpdir, "a.lean", "theorem foo : True := trivial")
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["name"], "foo")
        self.assertEqual(docs[0].metadata["kind"], "theorem")

    def test_lemma(self):
        path = _write(self.tmpdir, "a.lean", "lemma bar : 1 = 1 := rfl")
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["kind"], "lemma")
        self.assertEqual(docs[0].metadata["name"], "bar")

    def test_def(self):
        path = _write(self.tmpdir, "a.lean", "def baz : Nat := 42")
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["kind"], "def")
        self.assertEqual(docs[0].metadata["name"], "baz")

    def test_noncomputable_def(self):
        path = _write(self.tmpdir, "a.lean", "noncomputable def qux : Real := 3.14")
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["kind"], "noncomputable def")
        self.assertEqual(docs[0].metadata["name"], "qux")

    def test_multi_line_signature(self):
        content = (
            "theorem mul_comm\n"
            "  (a b : Nat) :\n"
            "  a * b = b * a := by ring"
        )
        path = _write(self.tmpdir, "a.lean", content)
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["name"], "mul_comm")
        # Signature should be flattened to a single line.
        self.assertNotIn("\n", docs[0].page_content.split(" : ", 1)[1])

    def test_with_docstring(self):
        content = (
            "/-- The identity theorem. -/\n"
            "theorem id_thm : True := trivial"
        )
        path = _write(self.tmpdir, "a.lean", content)
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertIn("The identity theorem.", docs[0].page_content)
        self.assertEqual(docs[0].metadata["name"], "id_thm")

    def test_with_attribute(self):
        content = "@[simp]\ntheorem simp_thm : 1 = 1 := rfl"
        path = _write(self.tmpdir, "a.lean", content)
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["kind"], "theorem")
        self.assertEqual(docs[0].metadata["name"], "simp_thm")

    def test_mixed_declarations(self):
        content = (
            "theorem t1 : True := trivial\n"
            "theorem t2 : True := trivial\n"
            "theorem t3 : True := trivial\n"
            "lemma l1 : 1 = 1 := rfl\n"
            "lemma l2 : 1 = 1 := rfl\n"
            "def d1 : Nat := 42\n"
        )
        path = _write(self.tmpdir, "a.lean", content)
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 6)
        kinds = [d.metadata["kind"] for d in docs]
        self.assertEqual(kinds.count("theorem"), 3)
        self.assertEqual(kinds.count("lemma"), 2)
        self.assertEqual(kinds.count("def"), 1)

    def test_empty_file(self):
        path = _write(self.tmpdir, "a.lean", "")
        docs = _parse_lean_file(path)
        self.assertEqual(docs, [])

    def test_only_comments(self):
        content = "-- foo\n/- bar -/\n"
        path = _write(self.tmpdir, "a.lean", content)
        docs = _parse_lean_file(path)
        self.assertEqual(docs, [])

    def test_missing_file_returns_empty(self):
        docs = _parse_lean_file(os.path.join(self.tmpdir, "does_not_exist.lean"))
        self.assertEqual(docs, [])

    def test_line_metadata_is_one_based(self):
        content = "\n\ntheorem foo : True := trivial\n"
        path = _write(self.tmpdir, "a.lean", content)
        docs = _parse_lean_file(path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["line"], 3)

    def test_file_metadata_preserved(self):
        path = _write(self.tmpdir, "a.lean", "theorem foo : True := trivial")
        docs = _parse_lean_file(path)
        self.assertEqual(docs[0].metadata["file"], path)


class FindMathlibRootTest(unittest.TestCase):
    """`_find_mathlib_root` returns None when no candidate matches."""

    @staticmethod
    def _make_dead_path():
        """Build a Path mock whose every probe resolves to a non-existent thing."""
        fake_inst = mock.MagicMock()
        fake_inst.resolve.return_value = fake_inst
        fake_inst.parent = fake_inst
        fake_inst.__truediv__ = lambda self, other: self
        fake_inst.exists.return_value = False
        # Some branches call `.iterdir()` on a Path; yield nothing.
        fake_inst.iterdir.return_value = iter([])
        return fake_inst

    @staticmethod
    def _block_lean_interact_import():
        """
        Return a context manager that makes `from lean_interact.config import ...`
        raise ImportError, even if lean_interact is already in sys.modules.
        """
        import builtins
        real_import = builtins.__import__

        # Stash + drop any cached lean_interact modules so __import__ runs fresh.
        cached = {k: sys.modules.pop(k) for k in list(sys.modules)
                  if k == "lean_interact" or k.startswith("lean_interact.")}

        def fake_import(name, *args, **kwargs):
            if name == "lean_interact" or name.startswith("lean_interact."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        patcher = mock.patch.object(builtins, "__import__", side_effect=fake_import)

        class _Ctx:
            def __enter__(self):
                patcher.start()
                return self

            def __exit__(self, *exc):
                patcher.stop()
                # Restore anything we stashed so other tests aren't affected.
                sys.modules.update(cached)

        return _Ctx()

    def test_returns_none_when_no_candidates_match(self):
        # Stub every filesystem probe so nothing resolves to a real Mathlib tree.
        # We also force the `lean_interact.config` import to fail so the
        # tmp_projects cache branch doesn't surface a real install on dev
        # machines that happen to have lean_interact installed.
        fake_inst = self._make_dead_path()
        with self._block_lean_interact_import(), \
             mock.patch("mathlib_corpus.Path", return_value=fake_inst), \
             mock.patch("mathlib_corpus.os.path.isdir", return_value=False), \
             mock.patch("mathlib_corpus.os.walk", return_value=iter([])):
            self.assertIsNone(_find_mathlib_root())

    def test_returns_none_when_isdir_false_everywhere(self):
        # Even if Path probes don't match, the os.walk loop is guarded by isdir.
        fake_inst = self._make_dead_path()
        with self._block_lean_interact_import(), \
             mock.patch("mathlib_corpus.os.path.isdir", return_value=False), \
             mock.patch("mathlib_corpus.Path", return_value=fake_inst):
            self.assertIsNone(_find_mathlib_root())


class MathLibCorpusExtractTest(unittest.TestCase):
    """End-to-end tests for `MathLibCorpus.extract`."""

    def test_raises_when_root_cannot_be_found(self):
        with mock.patch("mathlib_corpus._find_mathlib_root", return_value=None):
            corpus = MathLibCorpus()
            with self.assertRaises(RuntimeError) as ctx:
                corpus.extract()
            self.assertIn("Mathlib", str(ctx.exception))

    def test_extract_combines_docs_from_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "a.lean", "theorem t1 : True := trivial")
            _write(tmp, "b.lean", "lemma l1 : 1 = 1 := rfl")
            _write(tmp, "c.lean", "def d1 : Nat := 42")
            corpus = MathLibCorpus(mathlib_root=tmp)
            docs = corpus.extract()
            self.assertEqual(len(docs), 3)
            names = sorted(d.metadata["name"] for d in docs)
            self.assertEqual(names, ["d1", "l1", "t1"])

    def test_max_files_limits_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "a.lean", "theorem t1 : True := trivial")
            _write(tmp, "b.lean", "theorem t2 : True := trivial")
            _write(tmp, "c.lean", "theorem t3 : True := trivial")
            corpus = MathLibCorpus(mathlib_root=tmp)
            docs = corpus.extract(max_files=1)
            self.assertEqual(len(docs), 1)

    def test_prefers_mathlib_subdir_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            # File directly under tmp -- should be IGNORED when Mathlib/ exists.
            _write(tmp, "ignored.lean", "theorem ignored : True := trivial")
            mathlib_subdir = Path(tmp) / "Mathlib"
            mathlib_subdir.mkdir()
            _write(str(mathlib_subdir), "real.lean", "theorem real_thm : True := trivial")

            corpus = MathLibCorpus(mathlib_root=tmp)
            docs = corpus.extract()
            names = [d.metadata["name"] for d in docs]
            self.assertIn("real_thm", names)
            self.assertNotIn("ignored", names)

    def test_init_uses_find_mathlib_root_when_none_passed(self):
        with mock.patch("mathlib_corpus._find_mathlib_root", return_value="/fake/root"):
            corpus = MathLibCorpus()
            self.assertEqual(corpus.mathlib_root, "/fake/root")

    def test_init_respects_explicit_root(self):
        corpus = MathLibCorpus(mathlib_root="/explicit/root")
        self.assertEqual(corpus.mathlib_root, "/explicit/root")


if __name__ == "__main__":
    unittest.main()
