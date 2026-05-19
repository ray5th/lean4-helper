"""
Fuzz tests for the LLM output parser helpers in src/langgraph_agent.py.

These tests cover edge cases for:
    _extract_lean_code  — pull Lean source out of a ```lean fenced block
    _sanitize_imports   — force a single `import Mathlib`
    _count_theorem_blocks — count Lean declarations in a snippet

Bug found and fixed during fuzzing:
    `_count_theorem_blocks` previously used naive `startswith` matches on the
    tuple ("example", "theorem ", "lemma ", "def "). This meant:
      * `theorem\n` (newline immediately after the keyword, no space) was
        counted as 0 — inconsistent with `example` which had no space suffix.
      * `examplelike := rfl` was counted as 1 (false positive), because
        `example` had no trailing-space guard.
    The function now uses a regex that requires the keyword be followed by
    whitespace, end-of-line, or one of `:` / `(`, which fixes both issues.
"""
import os
import sys
import unittest

sys.path.insert(0, "src")
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from langgraph_agent import (  # noqa: E402
    _count_theorem_blocks,
    _extract_lean_code,
    _sanitize_imports,
)


class TestExtractLeanCode(unittest.TestCase):
    """Fuzz coverage for _extract_lean_code."""

    def test_empty_string(self):
        self.assertEqual(_extract_lean_code(""), "")

    def test_whitespace_only(self):
        self.assertEqual(_extract_lean_code("  \n  "), "")

    def test_no_fence_at_all(self):
        # No code fence -> return the text stripped.
        self.assertEqual(
            _extract_lean_code("theorem foo := rfl"),
            "theorem foo := rfl",
        )

    def test_no_fence_strips_whitespace(self):
        self.assertEqual(_extract_lean_code("\n  theorem foo := rfl  \n"),
                         "theorem foo := rfl")

    def test_plain_fence_no_lean_tag(self):
        # ``` ... ``` without the `lean` tag should still extract.
        self.assertEqual(_extract_lean_code("```\nfoo\n```"), "foo")

    def test_lean_fence_single_block(self):
        self.assertEqual(
            _extract_lean_code("```lean\ntheorem foo := rfl\n```"),
            "theorem foo := rfl",
        )

    def test_multiple_lean_blocks_keeps_first(self):
        # When the LLM emits multiple ```lean blocks, take the first one.
        text = "```lean\nfoo\n```\n\n```lean\nbar\n```"
        self.assertEqual(_extract_lean_code(text), "foo")

    def test_lean_fence_prefers_lean_tag_over_bare(self):
        # If both ```lean and a bare ``` are present, ```lean wins.
        text = "preamble ``` not-lean ```\n\n```lean\nfoo\n```"
        self.assertEqual(_extract_lean_code(text), "foo")

    def test_nested_backticks_in_content_truncates(self):
        # Known limitation: a nested ``` inside the lean block terminates
        # extraction early. We lock in the current behavior — extending the
        # parser to handle this would require a real tokenizer, and the
        # truncated output still fails Lean verification loudly.
        text = "```lean\nfoo ``` bar\n```"
        self.assertEqual(_extract_lean_code(text), "foo")

    def test_crlf_line_endings(self):
        self.assertEqual(
            _extract_lean_code("```lean\r\nfoo\r\n```"),
            "foo",
        )

    def test_trailing_whitespace_after_closing_fence(self):
        self.assertEqual(
            _extract_lean_code("```lean\nfoo\n```   \n  \t\n"),
            "foo",
        )

    def test_leading_text_before_fence(self):
        # LLMs love to chat before the code block.
        text = "Sure! Here is the proof:\n\n```lean\ntheorem foo := rfl\n```"
        self.assertEqual(_extract_lean_code(text), "theorem foo := rfl")

    def test_unicode_in_code(self):
        text = "```lean\n∀ n : ℕ, n = n\n```"
        self.assertEqual(_extract_lean_code(text), "∀ n : ℕ, n = n")

    def test_returns_str_always(self):
        # Robustness: every input should yield a string, never raise.
        for inp in ["", "  ", "x", "```", "```lean", "```lean\n```",
                    "```\n```", "\n\n\n", "```lean\n```lean\n```\n```"]:
            with self.subTest(inp=repr(inp)):
                out = _extract_lean_code(inp)
                self.assertIsInstance(out, str)


class TestSanitizeImports(unittest.TestCase):
    """Fuzz coverage for _sanitize_imports."""

    def test_no_imports_prepends_mathlib(self):
        out = _sanitize_imports("theorem foo := rfl")
        self.assertTrue(out.startswith("import Mathlib\n"))
        self.assertIn("theorem foo := rfl", out)

    def test_empty_string(self):
        out = _sanitize_imports("")
        # Must still start with `import Mathlib`, no crash.
        self.assertTrue(out.startswith("import Mathlib\n"))

    def test_only_imports_keeps_just_mathlib(self):
        out = _sanitize_imports("import Foo\nimport Bar\nimport Baz.Quux")
        self.assertTrue(out.startswith("import Mathlib\n"))
        # No leftover hallucinated imports.
        self.assertNotIn("import Foo", out)
        self.assertNotIn("import Bar", out)
        self.assertNotIn("import Baz", out)

    def test_multiple_imports_collapse_to_single_mathlib(self):
        code = (
            "import Mathlib.Data.Nat.Basic\n"
            "import Mathlib.Tactic\n"
            "import Std\n"
            "\n"
            "theorem foo : 1 = 1 := rfl\n"
        )
        out = _sanitize_imports(code)
        # Exactly one `import` line should remain — `import Mathlib`.
        import_lines = [l for l in out.splitlines() if l.strip().startswith("import ")]
        self.assertEqual(import_lines, ["import Mathlib"])
        self.assertIn("theorem foo : 1 = 1 := rfl", out)

    def test_import_with_leading_whitespace_is_stripped(self):
        code = "    import Mathlib.Data.Nat.Basic\ntheorem foo := rfl"
        out = _sanitize_imports(code)
        self.assertNotIn("Mathlib.Data.Nat.Basic", out)
        self.assertTrue(out.startswith("import Mathlib\n"))
        self.assertIn("theorem foo := rfl", out)

    def test_import_with_tab_indent(self):
        code = "\timport Foo\ntheorem foo := rfl"
        out = _sanitize_imports(code)
        self.assertNotIn("import Foo", out)
        self.assertTrue(out.startswith("import Mathlib\n"))

    def test_import_inside_string_literal_known_false_positive(self):
        # Known limitation: a line that, when stripped, starts with `import `
        # gets dropped even if it's inside a string. This is an acceptable
        # trade-off — Lean source rarely has bare lines like `import X` inside
        # multi-line strings, and the cost of a false positive (regenerate)
        # is much lower than a false negative (hallucinated import compiled).
        code = '#eval "import Foo"\ntheorem foo := rfl'
        out = _sanitize_imports(code)
        # The eval-with-string line does NOT strip-start with "import " (it
        # starts with `#eval`), so it should be preserved.
        self.assertIn('#eval "import Foo"', out)
        self.assertIn("theorem foo := rfl", out)

    def test_always_prepends_mathlib(self):
        # Every input — no matter how degenerate — must start with `import Mathlib`.
        for inp in ["", "   ", "\n\n", "theorem foo := rfl",
                    "import Foo", "import Mathlib", "garbage"]:
            with self.subTest(inp=repr(inp)):
                out = _sanitize_imports(inp)
                self.assertTrue(out.startswith("import Mathlib"))


class TestCountTheoremBlocks(unittest.TestCase):
    """Fuzz coverage for _count_theorem_blocks."""

    def test_single_theorem_with_space(self):
        self.assertEqual(_count_theorem_blocks("theorem foo := rfl"), 1)

    def test_theorem_immediately_followed_by_newline(self):
        # Bug fix: previously this returned 0 because the keyword tuple
        # required a trailing space. Now matched via regex with EOL/whitespace.
        self.assertEqual(_count_theorem_blocks("theorem\nfoo := rfl"), 1)

    def test_mixed_declarations(self):
        code = (
            "example : 1 = 1 := rfl\n"
            "lemma foo : 2 = 2 := rfl\n"
            "def bar : Nat := 1\n"
            "theorem baz : 3 = 3 := rfl\n"
        )
        self.assertEqual(_count_theorem_blocks(code), 4)

    def test_indented_declaration(self):
        self.assertEqual(
            _count_theorem_blocks("    theorem foo := rfl"),
            1,
        )

    def test_indented_with_tabs(self):
        self.assertEqual(
            _count_theorem_blocks("\t\ttheorem foo := rfl"),
            1,
        )

    def test_theoremlike_not_matched(self):
        # `theoremlike` must NOT count as a `theorem` declaration.
        self.assertEqual(_count_theorem_blocks("theoremlike := rfl"), 0)

    def test_examplelike_not_matched(self):
        # Bug fix: previously `example` had no trailing-space guard so
        # `examplelike := rfl` was a false positive.
        self.assertEqual(_count_theorem_blocks("examplelike := rfl"), 0)

    def test_lemmalike_not_matched(self):
        self.assertEqual(_count_theorem_blocks("lemmalike := rfl"), 0)

    def test_deflike_not_matched(self):
        self.assertEqual(_count_theorem_blocks("deflike := rfl"), 0)

    def test_empty_string(self):
        self.assertEqual(_count_theorem_blocks(""), 0)

    def test_whitespace_only(self):
        self.assertEqual(_count_theorem_blocks("   \n\n   "), 0)

    def test_example_with_colon(self):
        # Lean syntax: `example : 1 = 1 := rfl` — keyword followed by space
        # and `:`. Should still count as 1.
        self.assertEqual(_count_theorem_blocks("example : 1 = 1 := rfl"), 1)

    def test_realistic_proof_file(self):
        code = (
            "import Mathlib\n"
            "\n"
            "theorem add_zero_left (n : Nat) : 0 + n = n := by\n"
            "  exact Nat.zero_add n\n"
            "\n"
            "lemma my_lemma : True := trivial\n"
            "\n"
            "-- this is a comment with the word theorem in it\n"
            "def helper : Nat := 0\n"
        )
        # 1 theorem + 1 lemma + 1 def = 3. The comment line must NOT count.
        self.assertEqual(_count_theorem_blocks(code), 3)


if __name__ == "__main__":
    unittest.main()
