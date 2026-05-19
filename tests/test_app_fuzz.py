"""Fuzz tests for app.solve_proof.

These tests exercise solve_proof with adversarial inputs (empty/whitespace, very
large payloads, unicode, non-ASCII bytes, missing env vars, agent failures) and
assert two invariants for every input:

  1. solve_proof never crashes - it always returns a 3-tuple of strings.
  2. The temp file it creates is always cleaned up, even on agent failure.

LangGraphAgent and gradio are stubbed in sys.modules before app is imported, so
no real LLM/Lean toolchain is needed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock


# --- sys.path bootstrap (required so app's `from langgraph_agent import ...`
# --- can resolve to the stub we install below).
sys.path.insert(0, ".")
sys.path.insert(0, "src")

# Also resolve repo root in case tests are run from elsewhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# --- Stub LangGraphAgent BEFORE importing app -------------------------------
class _StubAgent:
    """A swap-in replacement for LangGraphAgent.

    Tests set ``_StubAgent.next_result`` (a dict) or ``_StubAgent.next_exception``
    (an Exception class/instance) before calling solve_proof. ``last_filepath``
    captures the path the agent saw, which lets tests assert the temp file was
    deleted after solve_proof returned.
    """

    next_result: dict | None = None
    next_exception: BaseException | None = None
    last_filepath: str | None = None
    init_calls: list[tuple] = []

    def __init__(self, model_name: str, max_retries: int):
        type(self).init_calls.append((model_name, max_retries))

    def solve_file_detailed(self, filepath: str) -> dict:
        type(self).last_filepath = filepath
        if type(self).next_exception is not None:
            raise type(self).next_exception
        if type(self).next_result is None:
            raise AssertionError("test forgot to set _StubAgent.next_result")
        return type(self).next_result


def _install_stub_modules() -> None:
    """Install fake langgraph_agent + gradio modules so importing app succeeds."""
    lga_module = types.ModuleType("langgraph_agent")
    lga_module.LangGraphAgent = _StubAgent  # type: ignore[attr-defined]
    sys.modules["langgraph_agent"] = lga_module

    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")

        class _Blocks:
            def __init__(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def launch(self, *_, **__):
                return None

        class _Row(_Blocks):
            pass

        class _Column(_Blocks):
            pass

        class _Accordion(_Blocks):
            pass

        class _Widget:
            def __init__(self, *_, **__):
                pass

            def click(self, *_, **__):
                return None

        class _Themes:
            class Base:
                def __init__(self, *_, **__):
                    pass

            @staticmethod
            def GoogleFont(*_, **__):
                return None

        gr.Blocks = _Blocks
        gr.Row = _Row
        gr.Column = _Column
        gr.Accordion = _Accordion
        gr.Dropdown = _Widget
        gr.Slider = _Widget
        gr.Button = _Widget
        gr.Code = _Widget
        gr.Textbox = _Widget
        gr.HTML = _Widget
        gr.themes = _Themes  # type: ignore[attr-defined]
        sys.modules["gradio"] = gr


_install_stub_modules()

import app  # noqa: E402  (intentional: stubs must be installed first)


# Track every temp file path app ever creates.
_CREATED_TEMP_PATHS: list[str] = []

_real_named_temp_file = tempfile.NamedTemporaryFile


def _tracking_named_temp_file(*args, **kwargs):
    """Wrapper around tempfile.NamedTemporaryFile that records the path."""
    handle = _real_named_temp_file(*args, **kwargs)
    _CREATED_TEMP_PATHS.append(handle.name)
    return handle


class SolveProofFuzzTests(unittest.TestCase):
    """Each test enforces: 3-tuple return + temp file cleaned up."""

    def setUp(self) -> None:
        _CREATED_TEMP_PATHS.clear()
        _StubAgent.next_result = None
        _StubAgent.next_exception = None
        _StubAgent.last_filepath = None
        _StubAgent.init_calls = []
        # Default: GROQ_API_KEY is set so cases that don't care about env var
        # don't accidentally bail on the env-var check.
        self._env_patch = mock.patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"})
        self._env_patch.start()
        self._tempfile_patch = mock.patch(
            "app.tempfile.NamedTemporaryFile", side_effect=_tracking_named_temp_file
        )
        self._tempfile_patch.start()

    def tearDown(self) -> None:
        self._tempfile_patch.stop()
        self._env_patch.stop()

    # ---- helpers --------------------------------------------------------
    def _assert_three_tuple_of_strings(self, result) -> None:
        self.assertIsInstance(result, tuple, f"expected tuple, got {type(result)!r}")
        self.assertEqual(
            len(result), 3, f"expected 3-tuple, got len={len(result)}: {result!r}"
        )
        for i, item in enumerate(result):
            self.assertIsInstance(
                item, str, f"item {i} expected str, got {type(item)!r}"
            )

    def _assert_all_temp_files_cleaned(self) -> None:
        for path in _CREATED_TEMP_PATHS:
            self.assertFalse(
                os.path.exists(path),
                f"temp file leaked: {path!r} still exists after solve_proof",
            )

    # ---- corpus ---------------------------------------------------------
    def test_empty_string_returns_no_input(self):
        status, code, logs = app.solve_proof("", "llama-3.3-70b-versatile", 1)
        self._assert_three_tuple_of_strings((status, code, logs))
        self.assertIn("No input", status)
        self.assertEqual(code, "")
        self.assertEqual(logs, "")
        # No temp file should have been created for empty input.
        self.assertEqual(_CREATED_TEMP_PATHS, [])

    def test_whitespace_only_returns_no_input(self):
        status, code, logs = app.solve_proof("   \n\t  ", "llama-3.3-70b-versatile", 1)
        self._assert_three_tuple_of_strings((status, code, logs))
        self.assertIn("No input", status)
        self.assertEqual(code, "")
        self.assertEqual(logs, "")
        self.assertEqual(_CREATED_TEMP_PATHS, [])

    def test_100kb_string_does_not_crash(self):
        _StubAgent.next_result = {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }
        big_payload = "a" * 100_000
        result = app.solve_proof(big_payload, "llama-3.3-70b-versatile", 1)
        self._assert_three_tuple_of_strings(result)
        # Verified payload made it to disk (and back).
        self.assertEqual(len(result[1]), 100_000)
        self._assert_all_temp_files_cleaned()

    def test_unicode_heavy_input_encodes_utf8(self):
        unicode_code = (
            "theorem foo : ∀ n : ℕ, ∃ m : ℕ, n + m = m + n := by sorry"
        )
        _StubAgent.next_result = {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }
        result = app.solve_proof(unicode_code, "llama-3.3-70b-versatile", 1)
        self._assert_three_tuple_of_strings(result)
        # File should round-trip the unicode unchanged.
        self.assertEqual(result[1], unicode_code)
        self._assert_all_temp_files_cleaned()

    def test_missing_groq_api_key(self):
        # Stop the default env patch so we control env precisely.
        self._env_patch.stop()
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                result = app.solve_proof(
                    "theorem foo : True := trivial", "llama-3.3-70b-versatile", 1
                )
        finally:
            # Restart so tearDown can stop it cleanly.
            self._env_patch.start()

        self._assert_three_tuple_of_strings(result)
        self.assertIn("GROQ_API_KEY missing", result[0])
        # No temp file should have been created when env var is missing.
        self.assertEqual(_CREATED_TEMP_PATHS, [])

    def test_agent_raises_runtime_error_returns_error_and_cleans_up(self):
        _StubAgent.next_exception = RuntimeError("agent boom")
        result = app.solve_proof(
            "theorem foo : True := trivial", "llama-3.3-70b-versatile", 1
        )
        self._assert_three_tuple_of_strings(result)
        self.assertIn("Error", result[0])
        # Even though the agent raised, the temp file must be cleaned up.
        self.assertEqual(len(_CREATED_TEMP_PATHS), 1)
        self._assert_all_temp_files_cleaned()

    def test_agent_success_status(self):
        _StubAgent.next_result = {
            "success": True,
            "solved_at_attempt": 2,
            "total_attempts": 3,
        }
        result = app.solve_proof(
            "theorem foo : True := trivial", "llama-3.3-70b-versatile", 3
        )
        self._assert_three_tuple_of_strings(result)
        self.assertIn("Verified", result[0])
        self.assertIn("2", result[0])
        self.assertIn("3", result[0])
        self._assert_all_temp_files_cleaned()

    def test_agent_failure_status(self):
        _StubAgent.next_result = {
            "success": False,
            "solved_at_attempt": None,
            "total_attempts": 5,
        }
        result = app.solve_proof(
            "theorem foo : True := trivial", "llama-3.3-70b-versatile", 5
        )
        self._assert_three_tuple_of_strings(result)
        self.assertIn("No proof found", result[0])
        self.assertIn("5", result[0])
        self._assert_all_temp_files_cleaned()

    def test_non_ascii_bytes_string_does_not_crash(self):
        _StubAgent.next_result = {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }
        # These are valid Unicode code points U+0080..U+0082 - they encode to
        # multi-byte UTF-8 but ASCII codecs would reject them.
        weird_input = "\x80\x81\x82"
        result = app.solve_proof(weird_input, "llama-3.3-70b-versatile", 1)
        self._assert_three_tuple_of_strings(result)
        # Round-trip should preserve the bytes when written/read as UTF-8.
        self.assertEqual(result[1], weird_input)
        self._assert_all_temp_files_cleaned()

    def test_no_sorry_keyword_still_passes_through(self):
        _StubAgent.next_result = {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }
        # Note: no 'sorry' anywhere. solve_proof should not reject this.
        result = app.solve_proof(
            "theorem foo : True := trivial", "llama-3.3-70b-versatile", 1
        )
        self._assert_three_tuple_of_strings(result)
        self.assertIn("Verified", result[0])
        # Agent must actually have been called - i.e. no early rejection.
        self.assertIsNotNone(_StubAgent.last_filepath)
        self._assert_all_temp_files_cleaned()

    def test_temp_file_written_as_utf8_on_non_utf8_locale(self):
        """Even when locale encoding is ASCII, unicode input must round-trip.

        This guards against the platform-portability bug where Python's text-mode
        tempfile defaults to ``locale.getpreferredencoding(False)`` which on
        Windows / non-UTF-8 systems would crash on unicode input. solve_proof
        must request UTF-8 explicitly.
        """
        _StubAgent.next_result = {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }

        def ascii_default_named_temp_file(*args, **kwargs):
            """Mimic a system where the default encoding is ASCII.

            If the caller (solve_proof) does not pass ``encoding=`` explicitly,
            we force ASCII. If it does pass ``encoding=``, we honor it. This
            reproduces what would happen on Windows / non-UTF-8 locales.
            """
            kwargs.setdefault("encoding", "ascii")
            handle = _real_named_temp_file(*args, **kwargs)
            _CREATED_TEMP_PATHS.append(handle.name)
            return handle

        # Replace the tracking patch with the stricter one.
        self._tempfile_patch.stop()
        strict_patch = mock.patch(
            "app.tempfile.NamedTemporaryFile",
            side_effect=ascii_default_named_temp_file,
        )
        strict_patch.start()
        try:
            unicode_code = (
                "theorem foo : ∀ n : ℕ, ∃ m : ℕ, n + m = m + n := by sorry"
            )
            result = app.solve_proof(unicode_code, "llama-3.3-70b-versatile", 1)
            self._assert_three_tuple_of_strings(result)
            # The function must not have fallen into the error branch.
            self.assertNotIn("Error", result[0], f"unexpected error: {result[0]!r}")
            self.assertEqual(result[1], unicode_code)
        finally:
            strict_patch.stop()
            self._tempfile_patch.start()
        self._assert_all_temp_files_cleaned()

    def test_tempfile_creation_failure_does_not_NameError(self):
        """If tempfile.NamedTemporaryFile itself raises, solve_proof must still
        return a 3-tuple of strings (not blow up with NameError from the finally
        block trying to access an unbound ``tmp`` variable)."""

        def boom(*_, **__):
            raise OSError("simulated /tmp missing")

        self._tempfile_patch.stop()
        boom_patch = mock.patch(
            "app.tempfile.NamedTemporaryFile", side_effect=boom
        )
        boom_patch.start()
        try:
            result = app.solve_proof(
                "theorem foo : True := trivial", "llama-3.3-70b-versatile", 1
            )
        finally:
            boom_patch.stop()
            self._tempfile_patch.start()

        self._assert_three_tuple_of_strings(result)
        self.assertIn("Error", result[0])


if __name__ == "__main__":
    unittest.main()
