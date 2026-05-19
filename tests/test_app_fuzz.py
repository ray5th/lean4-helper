"""
Fuzz tests for `app.solve_proof`.

These tests exercise a fixed corpus of edge-case inputs against
`app.solve_proof` and assert:
  * every call returns a (status_html, final_code, logs) 3-tuple,
  * no call crashes (uncaught exceptions),
  * the right status messages are produced for each scenario,
  * the temp file `solve_proof` creates is always cleaned up,
    *even* when the underlying agent raises.

We mock `langgraph_agent.LangGraphAgent` BEFORE importing `app` so no Lean,
LLM, or FAISS infrastructure is needed.
"""

import io
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

# Make `app.py` and `src/` importable.
sys.path.insert(0, ".")
sys.path.insert(0, "src")


# ----- Mock LangGraphAgent BEFORE importing app -----------------------------
_fake_lg_module = mock.MagicMock()


class _DummyAgent:
    """Stand-in for LangGraphAgent. Configured per-test via class attributes."""

    # Per-test knobs.
    # If set, __init__ records (model_name, max_retries) here.
    init_log: list = []
    # If set, solve_file_detailed raises this instance instead of returning.
    raise_exc: BaseException | None = None
    # If set, this dict is returned from solve_file_detailed.
    result: dict = {
        "success": True,
        "solved_at_attempt": 1,
        "total_attempts": 1,
    }
    # Record temp file path the agent was passed, plus the bytes on disk.
    last_path: str | None = None
    last_bytes: bytes | None = None

    def __init__(self, model_name="x", max_retries=5, **kwargs):
        self.model_name = model_name
        self.max_retries = max_retries
        _DummyAgent.init_log.append((model_name, max_retries))

    def solve_file_detailed(self, file_path: str) -> dict:
        _DummyAgent.last_path = file_path
        try:
            with open(file_path, "rb") as f:
                _DummyAgent.last_bytes = f.read()
        except OSError:
            _DummyAgent.last_bytes = None

        if _DummyAgent.raise_exc is not None:
            raise _DummyAgent.raise_exc
        return _DummyAgent.result


_fake_lg_module.LangGraphAgent = _DummyAgent
sys.modules["langgraph_agent"] = _fake_lg_module

# Default to having a key set; individual tests override this.
os.environ.setdefault("GROQ_API_KEY", "test-key")

import app  # noqa: E402   (must come after the mock above)


# Sentinels for the fuzz corpus -----------------------------------------------
EMPTY = ""
WHITESPACE = "   \n\t  "
HUGE = "a" * 100_000
UNICODE = "theorem foo : ∀ n : ℕ, ∃ m : ℕ, n + m = m + n := by sorry"
NO_SORRY = "theorem foo : True := trivial"


def _wrap_namedtemp(captured: list[str], lock: threading.Lock):
    """Return a side_effect for `tempfile.NamedTemporaryFile` that records the
    name of every actual tempfile created so callers can assert cleanup."""
    real_ntf = tempfile.NamedTemporaryFile

    def factory(*args, **kwargs):
        f = real_ntf(*args, **kwargs)
        with lock:
            captured.append(f.name)
        return f

    return factory


class SolveProofFuzzTests(unittest.TestCase):
    """Each scenario from the fuzz corpus."""

    def setUp(self):
        # Reset the dummy agent's class-level knobs.
        _DummyAgent.init_log = []
        _DummyAgent.raise_exc = None
        _DummyAgent.result = {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }
        _DummyAgent.last_path = None
        _DummyAgent.last_bytes = None

        # Make sure GROQ_API_KEY is set unless a test clears it.
        os.environ["GROQ_API_KEY"] = "test-key"

        self._captured_paths: list[str] = []
        self._capture_lock = threading.Lock()
        self._ntf_patch = mock.patch.object(
            app.tempfile,
            "NamedTemporaryFile",
            side_effect=_wrap_namedtemp(self._captured_paths, self._capture_lock),
        )
        self._ntf_patch.start()

    def tearDown(self):
        self._ntf_patch.stop()
        # Best-effort: clean up anything the test leaked so /tmp doesn't fill.
        for path in self._captured_paths:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assert_triple(self, ret):
        self.assertIsInstance(ret, tuple, f"return is not tuple: {ret!r}")
        self.assertEqual(len(ret), 3, f"return is not length-3 tuple: {ret!r}")
        status, code, logs = ret
        self.assertIsInstance(status, str)
        self.assertIsInstance(code, str)
        self.assertIsInstance(logs, str)
        return status, code, logs

    def _assert_temp_cleaned(self):
        """Every tempfile we observed must be gone now."""
        for path in self._captured_paths:
            self.assertFalse(
                os.path.exists(path),
                f"tempfile leaked: {path} still exists",
            )

    # ------------------------------------------------------------------
    # Empty string -> "No input"
    # ------------------------------------------------------------------
    def test_empty_string_returns_no_input(self):
        ret = app.solve_proof(EMPTY, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("No input", status)
        # No tempfile should have been created at all for empty input.
        self.assertEqual(self._captured_paths, [])
        self.assertEqual(code, "")
        self.assertEqual(logs, "")

    # ------------------------------------------------------------------
    # Whitespace-only -> "No input"
    # ------------------------------------------------------------------
    def test_whitespace_only_returns_no_input(self):
        ret = app.solve_proof(WHITESPACE, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("No input", status)
        self.assertEqual(self._captured_paths, [])

    # ------------------------------------------------------------------
    # 100KB string must not OOM/crash.
    # ------------------------------------------------------------------
    def test_huge_input_does_not_crash(self):
        ret = app.solve_proof(HUGE, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        # Should pass through to the agent (which returns success).
        self.assertIn("Verified", status)
        # The agent must have seen the full 100k bytes.
        self.assertIsNotNone(_DummyAgent.last_bytes)
        self.assertEqual(len(_DummyAgent.last_bytes), 100_000)
        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # Unicode-heavy: must encode UTF-8 to temp file successfully.
    # ------------------------------------------------------------------
    def test_unicode_heavy_encodes_utf8(self):
        ret = app.solve_proof(UNICODE, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("Verified", status)

        # The bytes the agent saw must be the UTF-8 encoding of UNICODE.
        self.assertIsNotNone(_DummyAgent.last_bytes)
        self.assertEqual(_DummyAgent.last_bytes, UNICODE.encode("utf-8"))

        # The `code` returned (read back from disk) must round-trip.
        self.assertEqual(code, UNICODE)

        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # Unicode-heavy under simulated POSIX/ASCII locale.
    #
    # On systems where `locale.getpreferredencoding()` is ASCII (e.g. some
    # Docker containers, HuggingFace Spaces with PYTHONUTF8=0 + LC_ALL=C),
    # `open(path, "w")` without an explicit encoding will reject characters
    # like ∀, ℕ, ∃. The fix is to pass `encoding="utf-8"` explicitly when
    # writing the temp file and reading it back.
    #
    # We force ASCII mode by patching the builtins/tempfile callables that
    # `app.solve_proof` uses, defaulting `encoding="ascii"` whenever the
    # caller didn't specify one. If the implementation forgot to pass
    # encoding="utf-8", the unicode write/read will UnicodeEncode/DecodeError.
    # ------------------------------------------------------------------
    def test_unicode_heavy_under_ascii_locale(self):
        real_open = open
        real_ntf = tempfile.NamedTemporaryFile

        def ascii_open(*args, **kwargs):
            # In text mode (no "b"), default to ASCII if no encoding given.
            mode = kwargs.get("mode")
            if mode is None and len(args) >= 2:
                mode = args[1]
            if mode is None:
                mode = "r"
            if "b" not in mode and "encoding" not in kwargs:
                kwargs["encoding"] = "ascii"
            return real_open(*args, **kwargs)

        def ascii_ntf(*args, **kwargs):
            mode = kwargs.get("mode", "w+b")
            if "b" not in mode and "encoding" not in kwargs:
                kwargs["encoding"] = "ascii"
            return real_ntf(*args, **kwargs)

        with mock.patch.object(app.tempfile, "NamedTemporaryFile",
                               side_effect=ascii_ntf), \
             mock.patch("builtins.open", side_effect=ascii_open):
            ret = app.solve_proof(UNICODE, "llama-3.3-70b-versatile", 1)

        status, code, logs = self._assert_triple(ret)
        # Must NOT crash with a UnicodeError leaking through as "Error:".
        self.assertNotIn(
            "UnicodeEncodeError", status,
            f"solve_proof leaked UnicodeEncodeError under ASCII locale: {status!r}",
        )
        self.assertNotIn(
            "UnicodeDecodeError", status,
            f"solve_proof leaked UnicodeDecodeError under ASCII locale: {status!r}",
        )
        # Must produce a Verified status: the agent didn't fail, the encoding
        # plumbing must transparently use utf-8.
        self.assertIn(
            "Verified", status,
            f"solve_proof returned non-Verified status under ASCII locale: {status!r}",
        )
        self.assertEqual(code, UNICODE)
        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # GROQ_API_KEY unset -> "GROQ_API_KEY missing"
    # ------------------------------------------------------------------
    def test_missing_groq_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ret = app.solve_proof(NO_SORRY, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("GROQ_API_KEY missing", status)
        # No tempfile created -- function short-circuits.
        self.assertEqual(self._captured_paths, [])
        self.assertEqual(code, "")
        self.assertEqual(logs, "")

    # ------------------------------------------------------------------
    # Agent raises -> "Error:" status; temp file still cleaned up.
    # ------------------------------------------------------------------
    def test_agent_raises_runtime_error(self):
        _DummyAgent.raise_exc = RuntimeError("synthetic blow-up")
        ret = app.solve_proof(NO_SORRY, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("Error", status)
        self.assertIn("synthetic blow-up", status)
        self.assertEqual(code, "")
        # Most important: temp file must be cleaned up even on exception.
        self.assertEqual(len(self._captured_paths), 1)
        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # Agent reports success -> "Verified" status.
    # ------------------------------------------------------------------
    def test_agent_success(self):
        _DummyAgent.result = {
            "success": True,
            "solved_at_attempt": 2,
            "total_attempts": 3,
        }
        ret = app.solve_proof(NO_SORRY, "llama-3.3-70b-versatile", 3)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("Verified", status)
        self.assertIn("2", status)
        self.assertIn("3", status)
        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # Agent reports failure -> "No proof found" status.
    # ------------------------------------------------------------------
    def test_agent_failure(self):
        _DummyAgent.result = {
            "success": False,
            "solved_at_attempt": None,
            "total_attempts": 5,
        }
        ret = app.solve_proof(NO_SORRY, "llama-3.3-70b-versatile", 5)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("No proof found", status)
        self.assertIn("5", status)
        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # Code without `sorry` should still pass through unchanged.
    # ------------------------------------------------------------------
    def test_no_sorry_keyword_passes_through(self):
        ret = app.solve_proof(NO_SORRY, "llama-3.3-70b-versatile", 1)
        status, code, logs = self._assert_triple(ret)
        self.assertIn("Verified", status)
        self.assertEqual(code, NO_SORRY)
        # Bytes seen by agent must equal the input encoded as utf-8.
        self.assertEqual(_DummyAgent.last_bytes, NO_SORRY.encode("utf-8"))
        self._assert_temp_cleaned()

    # ------------------------------------------------------------------
    # Sanity: tempfile cleanup verified for every scenario in one place.
    # ------------------------------------------------------------------
    def test_temp_file_always_cleaned_up(self):
        """Run every non-short-circuit scenario and assert no tempfile leaks."""
        scenarios = [
            ("ok", NO_SORRY, None),
            ("huge", HUGE, None),
            ("unicode", UNICODE, None),
            ("raise", NO_SORRY, RuntimeError("boom")),
        ]
        for label, code_in, exc in scenarios:
            _DummyAgent.raise_exc = exc
            with self.subTest(label):
                app.solve_proof(code_in, "llama-3.3-70b-versatile", 1)
            _DummyAgent.raise_exc = None
        self._assert_temp_cleaned()


if __name__ == "__main__":
    unittest.main()
