"""
Parse-test every problems/*.lean file through the agent's pure-Python
preprocessing helpers in src/langgraph_agent.py.

No Lean install required - these tests only exercise text-level helpers:
    - UTF-8 load
    - _count_theorem_blocks
    - _sanitize_imports (must preserve theorem block count)
    - _extract_lean_code (idempotent on raw .lean files with no fences)
"""

import os
import sys
import unittest
from pathlib import Path

# Add src to Python path
sys.path.insert(0, 'src')
# Also support running from repo root regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from langgraph_agent import (  # noqa: E402
    _count_theorem_blocks,
    _extract_lean_code,
    _sanitize_imports,
)

_PROBLEMS_DIR = _REPO_ROOT / "problems"

_EXPECTED_FILES = [
    "simple_add.lean",
    "test_problem.lean",
    "lecture4_problems.lean",
    "lecture5_problems.lean",
    "lecture6_problems.lean",
    "lecture7_problems.lean",
]


def _existing_problem_files():
    """Return the list of problem files that actually exist on disk."""
    files = []
    for name in _EXPECTED_FILES:
        p = _PROBLEMS_DIR / name
        if p.is_file():
            files.append(p)
    # Also pick up anything else in problems/*.lean we don't know about.
    for p in sorted(_PROBLEMS_DIR.glob("*.lean")):
        if p not in files:
            files.append(p)
    return files


class TestProblemsParse(unittest.TestCase):
    """One subTest per file per assertion - failures clearly identify the file."""

    @classmethod
    def setUpClass(cls):
        cls.files = _existing_problem_files()

    def test_problems_dir_exists(self):
        self.assertTrue(
            _PROBLEMS_DIR.is_dir(),
            f"problems/ directory not found at {_PROBLEMS_DIR}",
        )

    def test_utf8_load(self):
        """Each .lean file must be readable as UTF-8."""
        for path in self.files:
            with self.subTest(file=path.name):
                # Must not raise.
                content = path.read_text(encoding="utf-8")
                self.assertIsInstance(content, str)

    def test_theorem_block_count_positive(self):
        """_count_theorem_blocks must return >= 1 for every problem file."""
        for path in self.files:
            with self.subTest(file=path.name):
                content = path.read_text(encoding="utf-8")
                count = _count_theorem_blocks(content)
                self.assertGreaterEqual(
                    count,
                    1,
                    f"{path.name}: expected >= 1 theorem block, got {count}",
                )

    def test_sanitize_imports_preserves_theorem_count(self):
        """_sanitize_imports must not drop any theorem/example/lemma/def statements."""
        for path in self.files:
            with self.subTest(file=path.name):
                content = path.read_text(encoding="utf-8")
                original = _count_theorem_blocks(content)
                sanitized = _sanitize_imports(content)
                after = _count_theorem_blocks(sanitized)
                self.assertEqual(
                    after,
                    original,
                    f"{path.name}: _sanitize_imports changed block count "
                    f"({original} -> {after})",
                )

    def test_extract_lean_code_idempotent_on_raw_files(self):
        """Raw .lean files have no code fences, so _extract_lean_code(content)
        should fall through to content.strip()."""
        for path in self.files:
            with self.subTest(file=path.name):
                content = path.read_text(encoding="utf-8")
                # Raw .lean files should not contain markdown code fences.
                self.assertNotIn(
                    "```",
                    content,
                    f"{path.name}: unexpected ``` fence in raw .lean file",
                )
                extracted = _extract_lean_code(content)
                self.assertEqual(
                    extracted,
                    content.strip(),
                    f"{path.name}: _extract_lean_code not idempotent on raw file",
                )


# Skip individual expected files if missing (defensive — they all exist today).
for _name in _EXPECTED_FILES:
    _path = _PROBLEMS_DIR / _name

    def _make_test(p):
        @unittest.skipUnless(p.is_file(), f"{p.name} not present")
        def _test(self):
            content = p.read_text(encoding="utf-8")
            self.assertGreaterEqual(_count_theorem_blocks(content), 1)
            self.assertEqual(
                _count_theorem_blocks(_sanitize_imports(content)),
                _count_theorem_blocks(content),
            )
            self.assertEqual(_extract_lean_code(content), content.strip())
        return _test

    setattr(
        TestProblemsParse,
        f"test_file_{_name.replace('.', '_')}",
        _make_test(_path),
    )


if __name__ == "__main__":
    unittest.main()
