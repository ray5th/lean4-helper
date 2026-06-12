"""
Regression tests for bugs found in a manual code review (no agent involved).

Each test locks in a specific fix; if any of these start failing, a previously
fixed bug has been reintroduced.
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, "src")

# Other test files (test_app_fuzz, test_concurrent_solve) install a MagicMock
# in `sys.modules["langgraph_agent"]`, which would shadow the real module and
# make our helpers return MagicMocks instead of ints/strings. Evict the cached
# mock so we get a fresh, real import.
sys.modules.pop("langgraph_agent", None)

# `langgraph_agent` imports lean_verifier + retriever + rag_chain which each
# pull in heavyweight ML libs. Mock those three modules just long enough to
# import the pure helpers, then RESTORE sys.modules — leaving MagicMocks
# behind poisons every test module imported after this one during discovery.
_DEPS = ("lean_verifier", "retriever", "rag_chain")
_saved = {m: sys.modules.get(m) for m in _DEPS}
for _mod in _DEPS:
    sys.modules[_mod] = mock.MagicMock()
try:
    import langgraph_agent

    importlib.reload(langgraph_agent)  # ensure regex/keyword changes are picked up
    _count_theorem_blocks = langgraph_agent._count_theorem_blocks
    _extract_lean_code = langgraph_agent._extract_lean_code
    _read_file = langgraph_agent._read_file
    _write_file = langgraph_agent._write_file
finally:
    for _mod, _orig in _saved.items():
        if _orig is not None:
            sys.modules[_mod] = _orig
        else:
            sys.modules.pop(_mod, None)
    # This langgraph_agent was built against mocks — evict it so later
    # importers get a real one.
    sys.modules.pop("langgraph_agent", None)


class TestCountTheoremBlocksKeywordFix(unittest.TestCase):
    """
    The old `_THEOREM_KEYWORDS = ("example", "theorem ", "lemma ", "def ")`
    tuple had inconsistent trailing spaces:
      - `theorem\nfoo` returned 0 (false negative — `theorem ` requires a space)
      - `examplelike` returned 1 (false positive — `example` has no boundary)
    Fixed by switching to a regex with a proper boundary class.
    """

    def test_theorem_followed_by_newline_counts(self):
        self.assertEqual(_count_theorem_blocks("theorem\nfoo := rfl"), 1)

    def test_examplelike_does_not_match(self):
        self.assertEqual(_count_theorem_blocks("examplelike := 5"), 0)

    def test_indented_theorem_counts(self):
        self.assertEqual(_count_theorem_blocks("    theorem foo : True := trivial"), 1)

    def test_theorem_with_colon_immediately(self):
        # `theorem:` (no space) should NOT match — only valid Lean syntax counts.
        self.assertEqual(_count_theorem_blocks("theorem:bar"), 0)

    def test_theorem_with_paren_immediately_does_NOT_count(self):
        # Not valid Lean — declaration name must come before `(`.
        self.assertEqual(_count_theorem_blocks("theorem(x : Nat) : True := trivial"), 0)

    def test_mixed_keywords_each_count(self):
        code = (
            "example : True := trivial\n"
            "theorem foo : True := trivial\n"
            "lemma bar : True := trivial\n"
            "def baz : Nat := 0\n"
        )
        self.assertEqual(_count_theorem_blocks(code), 4)


class TestExtractLeanCodeFenceVariants(unittest.TestCase):
    r"""
    `_extract_lean_code` used to do `text.split("```lean")[1]` which would
    leave a `4` at the start when the fence was `\`\`\`lean4` (a common LLM
    output). Now uses a regex that requires the lean tag to be terminated by
    whitespace.
    """

    def test_lean_fence(self):
        self.assertEqual(
            _extract_lean_code("```lean\nfoo := rfl\n```"),
            "foo := rfl",
        )

    def test_lean4_fence_does_not_leak_digit(self):
        # Old behavior: returned "4\nfoo := rfl"
        self.assertEqual(
            _extract_lean_code("```lean4\nfoo := rfl\n```"),
            "foo := rfl",
        )

    def test_uppercase_lean_fence(self):
        self.assertEqual(
            _extract_lean_code("```Lean\nfoo := rfl\n```"),
            "foo := rfl",
        )

    def test_no_fence_returns_stripped(self):
        self.assertEqual(_extract_lean_code("  foo := rfl  "), "foo := rfl")

    def test_untagged_fence(self):
        self.assertEqual(_extract_lean_code("```\nfoo := rfl\n```"), "foo := rfl")

    def test_leanish_tag_falls_through_to_untagged(self):
        # `leanish` isn't a recognized Lean fence, so the function should fall
        # through to the untagged-fence branch. That branch returns the raw
        # fenced content as-is (including "leanish" on the first line) — Lean
        # will then reject it as a syntax error and we retry. The important
        # thing is no crash.
        result = _extract_lean_code("```leanish\nfoo\n```")
        self.assertIn("foo", result)


class TestUtf8FileIO(unittest.TestCase):
    """
    `_read_file` / `_write_file` previously used the platform default encoding,
    which is C/ASCII on minimal Docker images and would crash on Lean unicode.
    """

    def test_round_trip_unicode_lean(self):
        code = "theorem foo : ∀ n : ℕ, ∃ m, n + m = m + n := by sorry"
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w",
                                          delete=False, encoding="utf-8") as f:
            path = f.name
        try:
            _write_file(path, code)
            self.assertEqual(_read_file(path), code)
        finally:
            os.unlink(path)


# (A TestProofAgentDefaultModel regression test lived here until the unused
# `proof_agent` wrapper module it covered was deleted.)


if __name__ == "__main__":
    unittest.main()
