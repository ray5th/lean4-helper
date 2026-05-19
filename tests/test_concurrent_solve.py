"""
Stress tests for app.solve_proof under concurrent invocation.

These tests mock out LangGraphAgent before importing app, so no Lean/LLM/Groq
infrastructure is required. They focus on:
  * 10 simultaneous solve_proof calls returning distinct, non-cross-contaminated
    final_code values (one tempfile per call, paths unique)
  * Tempfile cleanup (no leftover .lean files in /tmp afterwards)
  * Exception isolation (one failing call must not affect the other 9)
  * Detecting the stdout-redirection race condition that the original
    contextlib.redirect_stdout based implementation exhibited.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import mock

# Make `app` and `src/` importable.
sys.path.insert(0, ".")
sys.path.insert(0, "src")

# IMPORTANT: mock LangGraphAgent BEFORE importing app so app.py never tries to
# instantiate the real agent (which would require lean, langchain, faiss, etc.).
_fake_lg_module = mock.MagicMock()


class _DummyAgent:
    """Stand-in for LangGraphAgent. Configured per-test via class attributes."""

    # Per-test knobs (mutated by tests before each scenario).
    sleep_seconds: float = 0.0
    # If a thread's lean_code contains this token, the agent raises.
    raise_token: str | None = None
    # Optional: tag to print so we can detect cross-thread stdout bleed.
    print_tag: bool = False

    def __init__(self, model_name="x", max_retries=5, **kwargs):
        self.model_name = model_name
        self.max_retries = max_retries

    def solve_file_detailed(self, file_path: str) -> dict:
        # Read what app.solve_proof just wrote so we can react per-call.
        with open(file_path, "r") as f:
            code = f.read()

        if self.print_tag:
            # Print enough to be racy if stdout is process-global.
            for _ in range(20):
                print(f"TAG::{code}")

        # Sleep so calls overlap in real wall time.
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)

        if self.raise_token and self.raise_token in code:
            raise RuntimeError(f"injected failure for {self.raise_token}")

        return {
            "success": True,
            "solved_at_attempt": 1,
            "total_attempts": 1,
        }


_fake_lg_module.LangGraphAgent = _DummyAgent
sys.modules["langgraph_agent"] = _fake_lg_module

# Pre-set GROQ_API_KEY so solve_proof doesn't short-circuit.
os.environ.setdefault("GROQ_API_KEY", "test-key")

import app  # noqa: E402   (must come after the mock above)


def _scan_tmp_lean(prefix_dir: str = "/tmp") -> set[str]:
    """Return absolute paths of all .lean files currently in `prefix_dir`."""
    if not os.path.isdir(prefix_dir):
        return set()
    return {
        os.path.join(prefix_dir, f)
        for f in os.listdir(prefix_dir)
        if f.endswith(".lean")
    }


class ConcurrentSolveTests(unittest.TestCase):
    def setUp(self):
        # Reset agent state between scenarios.
        _DummyAgent.sleep_seconds = 0.0
        _DummyAgent.raise_token = None
        _DummyAgent.print_tag = False
        # Per-test patch: `app.solve_proof` does `from langgraph_agent import
        # LangGraphAgent` at import time, so `app.LangGraphAgent` is bound to
        # whichever module-level mock loaded first. Other test files do the
        # same trick and can win the race, so we patch on `app` directly.
        self._lg_patcher = mock.patch.object(app, "LangGraphAgent", _DummyAgent)
        self._lg_patcher.start()
        self.addCleanup(self._lg_patcher.stop)
        # Don't construct a real LeanEnvironment / FAISS retriever in tests.
        self._components_patcher = mock.patch.object(
            app, "_get_components", return_value=(mock.MagicMock(), mock.MagicMock())
        )
        self._components_patcher.start()
        self.addCleanup(self._components_patcher.stop)
        # Snapshot pre-existing tmp .lean files so we ignore unrelated ones.
        self._tmp_before = _scan_tmp_lean()

    def tearDown(self):
        # Best-effort cleanup of anything the tests created that survived a bug.
        leaked = _scan_tmp_lean() - self._tmp_before
        for path in leaked:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 1 + 3: 10 parallel calls, agent sleeps so calls overlap.
    # ------------------------------------------------------------------
    def test_ten_parallel_calls_no_cross_contamination(self):
        _DummyAgent.sleep_seconds = 0.1  # force overlap

        inputs = [f"theorem t_{i} : True := trivial" for i in range(10)]

        def run(code):
            return code, app.solve_proof(code, "llama-3.3-70b-versatile", 1)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(run, code) for code in inputs]
            results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 10)

        for code, triple in results:
            self.assertIsInstance(triple, tuple, f"{code!r} did not return tuple")
            self.assertEqual(len(triple), 3, f"{code!r} returned {triple!r}")
            status_html, final_code, logs = triple
            # final_code is what the agent wrote/left in the tempfile, which the
            # mock leaves untouched, so it must equal the input verbatim.
            self.assertEqual(
                final_code,
                code,
                f"Cross-contamination: expected {code!r} got {final_code!r}",
            )
            self.assertIn("Verified", status_html, f"Unexpected status: {status_html!r}")

        # All tempfiles must be cleaned up.
        leaked = _scan_tmp_lean() - self._tmp_before
        self.assertEqual(leaked, set(), f"Tempfiles leaked: {leaked!r}")

    # ------------------------------------------------------------------
    # 2: Tempfile collision check. Patch NamedTemporaryFile to log paths.
    # ------------------------------------------------------------------
    def test_tempfile_paths_unique_under_concurrency(self):
        _DummyAgent.sleep_seconds = 0.05

        seen_paths: list[str] = []
        seen_lock = threading.Lock()

        real_ntf = tempfile.NamedTemporaryFile

        def logging_ntf(*args, **kwargs):
            f = real_ntf(*args, **kwargs)
            with seen_lock:
                seen_paths.append(f.name)
            return f

        with mock.patch.object(app.tempfile, "NamedTemporaryFile", side_effect=logging_ntf):
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = [
                    ex.submit(
                        app.solve_proof,
                        f"theorem unique_{i} : True := trivial",
                        "llama-3.3-70b-versatile",
                        1,
                    )
                    for i in range(10)
                ]
                results = [f.result() for f in as_completed(futures)]

        self.assertEqual(len(results), 10)
        self.assertEqual(
            len(seen_paths),
            len(set(seen_paths)),
            f"Tempfile path collision: {seen_paths!r}",
        )

    # ------------------------------------------------------------------
    # 4: one call raises, the other 9 still succeed.
    # ------------------------------------------------------------------
    def test_one_raising_call_does_not_break_others(self):
        _DummyAgent.sleep_seconds = 0.1
        _DummyAgent.raise_token = "BOOM_3"

        inputs = []
        for i in range(10):
            if i == 3:
                inputs.append("theorem t_BOOM_3 : True := trivial")
            else:
                inputs.append(f"theorem t_{i} : True := trivial")

        def run(code):
            return code, app.solve_proof(code, "llama-3.3-70b-versatile", 1)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(run, code) for code in inputs]
            results = [f.result() for f in as_completed(futures)]

        oks = []
        errs = []
        for code, triple in results:
            self.assertIsInstance(triple, tuple)
            self.assertEqual(len(triple), 3)
            status_html, final_code, logs = triple
            if "BOOM_3" in code:
                errs.append((code, status_html, final_code))
            else:
                oks.append((code, status_html, final_code))

        self.assertEqual(len(oks), 9, "Expected 9 successful calls")
        self.assertEqual(len(errs), 1, "Expected exactly 1 erroring call")

        # Erroring call returns the "✗ Error:" status (per app.solve_proof).
        _, err_status, err_final = errs[0]
        self.assertIn("Error", err_status)
        self.assertEqual(err_final, "")

        # The 9 successful calls keep their own inputs intact.
        for code, status, final in oks:
            self.assertEqual(final, code)
            self.assertIn("Verified", status)

        # No tempfile leaks even on the error path.
        leaked = _scan_tmp_lean() - self._tmp_before
        self.assertEqual(leaked, set(), f"Tempfiles leaked: {leaked!r}")

    # ------------------------------------------------------------------
    # Stdout race: with print() inside the agent, log buffers must NOT
    # see each other's output. This is the bug redirect_stdout causes.
    # ------------------------------------------------------------------
    def test_stdout_logs_are_not_cross_contaminated(self):
        _DummyAgent.sleep_seconds = 0.1
        _DummyAgent.print_tag = True  # agent prints "TAG::<code>" 20 times

        inputs = [f"theorem stdoutcheck_{i} : True := trivial" for i in range(10)]

        def run(code):
            return code, app.solve_proof(code, "llama-3.3-70b-versatile", 1)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(run, c) for c in inputs]
            results = [f.result() for f in as_completed(futures)]

        # Each call's `logs` should contain its own TAG line and only its own.
        offenders = []
        for code, triple in results:
            _, _, logs = triple
            own_tag = f"TAG::{code}"
            if own_tag not in logs:
                offenders.append(("own-tag-missing", code, logs[:200]))
                continue
            # No other call's code should leak into this log.
            for other in inputs:
                if other == code:
                    continue
                if f"TAG::{other}" in logs:
                    offenders.append(("other-tag-leaked", code, other))
                    break

        self.assertEqual(
            offenders,
            [],
            "Stdout was cross-contaminated between concurrent solve_proof calls "
            "(redirect_stdout is process-global, not thread-local). "
            f"Offenders: {offenders[:5]!r}",
        )


if __name__ == "__main__":
    unittest.main()
