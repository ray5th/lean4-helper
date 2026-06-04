"""Unit tests for pure helpers in scripts/benchmark.py.

Covered helpers:
    - load_local_problems(problems_dir, max_problems)
    - pass_at_k(results, k)
    - _ensure_import_and_sorry(code)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Stub out the heavy `langgraph_agent` import so `benchmark` loads without
# its full dependency chain (langchain_anthropic, etc.).
sys.modules.setdefault(
    "langgraph_agent", mock.MagicMock(LangGraphAgent=mock.MagicMock())
)

from benchmark import _ensure_import_and_sorry, load_local_problems, pass_at_k  # noqa: E402

# ---------------------------------------------------------------------------
# load_local_problems
# ---------------------------------------------------------------------------

class LoadLocalProblemsTests(unittest.TestCase):
    def test_filters_files_without_sorry(self):
        """A complete (no sorry) file is skipped; a withsorry file is loaded."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "complete.lean").write_text(
                "import Mathlib\n\ntheorem foo : 1 + 1 = 2 := by rfl\n",
                encoding="utf-8",
            )
            (root / "withsorry.lean").write_text(
                "import Mathlib\n\ntheorem bar : 1 + 1 = 2 := by sorry\n",
                encoding="utf-8",
            )
            rows = load_local_problems(td)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "withsorry")

    def test_max_problems_limits_loading(self):
        """max_problems=2 with 5 sorry-files yields exactly the first 2 sorted."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            names = ["e_fifth", "a_first", "b_second", "c_third", "d_fourth"]
            for n in names:
                (root / f"{n}.lean").write_text(
                    f"-- {n}\nimport Mathlib\nexample : True := by sorry\n",
                    encoding="utf-8",
                )
            rows = load_local_problems(td, max_problems=2)
            self.assertEqual(len(rows), 2)
            # Sorted alphabetically: a_first, b_second are the first 2
            self.assertEqual(rows[0]["name"], "a_first")
            self.assertEqual(rows[1]["name"], "b_second")

    def test_max_problems_none_loads_all(self):
        """max_problems=None loads all sorry-files."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for n in ["p1", "p2", "p3"]:
                (root / f"{n}.lean").write_text(
                    "import Mathlib\nexample : True := by sorry\n",
                    encoding="utf-8",
                )
            rows = load_local_problems(td, max_problems=None)
            self.assertEqual(len(rows), 3)

    def test_missing_dir_raises_runtime_error(self):
        """Non-existent directory raises RuntimeError with a clear message."""
        missing = "/tmp/definitely_does_not_exist_xyz_12345abc"
        with self.assertRaises(RuntimeError) as ctx:
            load_local_problems(missing)
        self.assertIn(missing, str(ctx.exception))

    def test_empty_dir_returns_empty_list(self):
        """An empty directory returns [] without crashing."""
        with tempfile.TemporaryDirectory() as td:
            rows = load_local_problems(td)
            self.assertEqual(rows, [])

    def test_name_is_file_stem(self):
        """`name` is the file stem (no .lean suffix)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "lecture4_1.lean").write_text(
                "import Mathlib\nexample : True := by sorry\n",
                encoding="utf-8",
            )
            rows = load_local_problems(td)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "lecture4_1")

    def test_ignores_non_lean_files(self):
        """Non-.lean files (e.g. notes.txt) are ignored even if they contain 'sorry'."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "notes.txt").write_text(
                "I'm sorry to say this is not Lean.\n", encoding="utf-8"
            )
            (root / "prob.lean").write_text(
                "import Mathlib\nexample : True := by sorry\n",
                encoding="utf-8",
            )
            rows = load_local_problems(td)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "prob")


# ---------------------------------------------------------------------------
# pass_at_k
# ---------------------------------------------------------------------------

class PassAtKTests(unittest.TestCase):
    def test_empty_results(self):
        self.assertEqual(pass_at_k([], 1), 0.0)
        self.assertEqual(pass_at_k([], 5), 0.0)

    def test_all_success_attempt_one(self):
        results = [
            {"success": True, "solved_at_attempt": 1} for _ in range(5)
        ]
        self.assertEqual(pass_at_k(results, 1), 1.0)

    def test_three_success_at_attempt_two_plus_two_fail(self):
        results = (
            [{"success": True, "solved_at_attempt": 2} for _ in range(3)]
            + [{"success": False, "solved_at_attempt": None} for _ in range(2)]
        )
        self.assertAlmostEqual(pass_at_k(results, 2), 0.6)
        self.assertEqual(pass_at_k(results, 1), 0.0)

    def test_success_at_attempt_three_k_two(self):
        results = [{"success": True, "solved_at_attempt": 3}]
        self.assertEqual(pass_at_k(results, 2), 0.0)

    def test_success_at_attempt_three_k_ten(self):
        results = [
            {"success": True, "solved_at_attempt": 3},
            {"success": False, "solved_at_attempt": None},
            {"success": False, "solved_at_attempt": None},
            {"success": False, "solved_at_attempt": None},
        ]
        # 1 solved out of 4 total
        self.assertEqual(pass_at_k(results, 10), 0.25)

    def test_all_failures(self):
        results = [
            {"success": False, "solved_at_attempt": None} for _ in range(4)
        ]
        for k in (1, 2, 5, 100):
            self.assertEqual(pass_at_k(results, k), 0.0)


# ---------------------------------------------------------------------------
# _ensure_import_and_sorry
# ---------------------------------------------------------------------------

class EnsureImportAndSorryTests(unittest.TestCase):
    def test_missing_import_prepends_mathlib(self):
        code = "theorem foo : 1 + 1 = 2 := by sorry"
        out = _ensure_import_and_sorry(code)
        self.assertTrue(out.startswith("import Mathlib\n\n"))

    def test_has_import_and_sorry_unchanged(self):
        code = "import Mathlib\n\ntheorem foo : 1 + 1 = 2 := by sorry\n"
        out = _ensure_import_and_sorry(code)
        self.assertEqual(out, code)

    def test_has_import_missing_sorry_ends_by(self):
        code = "import Mathlib\n\ntheorem foo : 1 + 1 = 2 := by"
        out = _ensure_import_and_sorry(code)
        self.assertIn("sorry", out)
        # Should append sorry after "by"
        self.assertTrue(out.rstrip().endswith("sorry"))

    def test_has_import_missing_sorry_no_by(self):
        code = "import Mathlib\n\ntheorem foo : 1 + 1 = 2 :="
        out = _ensure_import_and_sorry(code)
        self.assertIn("sorry", out)
        self.assertIn("by", out)
        # Should append " by\n  sorry\n"
        self.assertTrue(out.rstrip().endswith("sorry"))

    def test_empty_string_no_crash_locks_current_behavior(self):
        """Empty input must not crash. Lock in observed output."""
        out = _ensure_import_and_sorry("")
        # No import in "" → prepended. No ":= by" / ":=" → nothing else appended.
        self.assertEqual(out, "import Mathlib\n\n")


if __name__ == "__main__":
    unittest.main()
