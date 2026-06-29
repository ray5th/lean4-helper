"""
Tests for app._get_components caching + thread safety.

`_get_components` is a lazy initializer that should:
  * construct LeanEnvironment and MathLibRetriever exactly once per process
  * cache the resulting instances in module-level globals
  * be safe to call from multiple threads (a lock must prevent double-init)

These tests patch the heavyweight classes at app-module level so no real Lean
REPL / FAISS index is loaded.
"""

import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

# Make `app` and `src/` importable regardless of cwd.
sys.path.insert(0, ".")
sys.path.insert(0, "src")

# Mock the heavyweight modules BEFORE importing app so module-level execution
# never tries to load real Lean / FAISS code.
sys.modules.setdefault("langgraph_agent", mock.MagicMock())
sys.modules.setdefault("lean_verifier", mock.MagicMock())
sys.modules.setdefault("retriever", mock.MagicMock())

import app  # noqa: E402   (must come after the mocks above)


class GetComponentsCacheTests(unittest.TestCase):
    def setUp(self):
        # Reset module-level cache so each test starts from a cold state.
        app._LEAN_ENV = None
        app._RETRIEVER = None

    def tearDown(self):
        # Same on the way out, so a failing test cannot poison the next one.
        app._LEAN_ENV = None
        app._RETRIEVER = None

    # ------------------------------------------------------------------
    # 1. First call constructs both LeanEnvironment and MathLibRetriever.
    # ------------------------------------------------------------------
    def test_first_call_constructs_both(self):
        with mock.patch.object(app, "LeanEnvironment") as fake_env, \
             mock.patch.object(app, "MathLibRetriever") as fake_retr:
            env, retr = app._get_components()

            self.assertEqual(
                fake_env.call_count, 1,
                f"LeanEnvironment should be constructed once, got {fake_env.call_count}",
            )
            self.assertEqual(
                fake_retr.call_count, 1,
                f"MathLibRetriever should be constructed once, got {fake_retr.call_count}",
            )
            # Sanity: returned values are what the mock produced.
            self.assertIs(env, fake_env.return_value)
            self.assertIs(retr, fake_retr.return_value)

    # ------------------------------------------------------------------
    # 2. Second call returns the SAME OBJECTS (cache hit, no re-init).
    # ------------------------------------------------------------------
    def test_second_call_returns_cached_instances(self):
        with mock.patch.object(app, "LeanEnvironment") as fake_env, \
             mock.patch.object(app, "MathLibRetriever") as fake_retr:
            env1, retr1 = app._get_components()
            env2, retr2 = app._get_components()

            # Identity, not equality.
            self.assertIs(env1, env2, "Second call returned a different LeanEnvironment instance")
            self.assertIs(retr1, retr2, "Second call returned a different MathLibRetriever instance")

            # And the classes were only invoked once total.
            self.assertEqual(
                fake_env.call_count, 1,
                f"LeanEnvironment constructed {fake_env.call_count} times across 2 calls",
            )
            self.assertEqual(
                fake_retr.call_count, 1,
                f"MathLibRetriever constructed {fake_retr.call_count} times across 2 calls",
            )

    # ------------------------------------------------------------------
    # 3. Thread safety: 10 concurrent callers, mock class constructed exactly once.
    #
    # We widen the race window by sleeping inside the mocked __init__ so
    # without the lock multiple threads would observe `_LEAN_ENV is None`
    # and re-enter the constructor.
    # ------------------------------------------------------------------
    def test_concurrent_callers_init_exactly_once(self):
        env_call_count = 0
        retr_call_count = 0
        count_lock = threading.Lock()

        def slow_env_init(*args, **kwargs):
            nonlocal env_call_count
            with count_lock:
                env_call_count += 1
            # Widen the race window: any unprotected `if _LEAN_ENV is None`
            # branch would let another thread reach the same `is None` check
            # while we're still here.
            time.sleep(0.05)
            return mock.MagicMock(name="LeanEnvInstance")

        def slow_retr_init(*args, **kwargs):
            nonlocal retr_call_count
            with count_lock:
                retr_call_count += 1
            time.sleep(0.05)
            return mock.MagicMock(name="RetrieverInstance")

        with mock.patch.object(app, "LeanEnvironment", side_effect=slow_env_init), \
             mock.patch.object(app, "MathLibRetriever", side_effect=slow_retr_init):
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = [ex.submit(app._get_components) for _ in range(10)]
                results = [f.result() for f in futures]

        self.assertEqual(
            env_call_count, 1,
            f"LeanEnvironment was constructed {env_call_count} times under "
            "concurrent access; the lock failed to serialize first-init.",
        )
        self.assertEqual(
            retr_call_count, 1,
            f"MathLibRetriever was constructed {retr_call_count} times under "
            "concurrent access; the lock failed to serialize first-init.",
        )
        self.assertEqual(len(results), 10)

    # ------------------------------------------------------------------
    # 4. All 10 threads receive the same (env, retriever) tuple by identity.
    # ------------------------------------------------------------------
    def test_concurrent_callers_get_identical_tuple(self):
        def slow_env_init(*args, **kwargs):
            time.sleep(0.05)
            return mock.MagicMock(name="LeanEnvInstance")

        def slow_retr_init(*args, **kwargs):
            time.sleep(0.05)
            return mock.MagicMock(name="RetrieverInstance")

        with mock.patch.object(app, "LeanEnvironment", side_effect=slow_env_init), \
             mock.patch.object(app, "MathLibRetriever", side_effect=slow_retr_init):
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = [ex.submit(app._get_components) for _ in range(10)]
                results = [f.result() for f in futures]

        self.assertEqual(len(results), 10)
        env_ids = {id(env) for env, _ in results}
        retr_ids = {id(retr) for _, retr in results}
        self.assertEqual(
            len(env_ids), 1,
            f"Threads received {len(env_ids)} distinct LeanEnvironment objects: {env_ids!r}",
        )
        self.assertEqual(
            len(retr_ids), 1,
            f"Threads received {len(retr_ids)} distinct MathLibRetriever objects: {retr_ids!r}",
        )

    # ------------------------------------------------------------------
    # 5. The module-level lock exists and is a real lock (has acquire/release).
    # ------------------------------------------------------------------
    def test_component_lock_is_a_lock(self):
        self.assertIsNotNone(app._COMPONENT_LOCK, "_COMPONENT_LOCK should not be None")
        # threading.Lock() returns a _thread.lock instance, which is the exact
        # type returned by `type(threading.Lock())`.
        self.assertIsInstance(app._COMPONENT_LOCK, type(threading.Lock()))
        # And it actually behaves like a lock.
        self.assertTrue(hasattr(app._COMPONENT_LOCK, "acquire"))
        self.assertTrue(hasattr(app._COMPONENT_LOCK, "release"))


if __name__ == "__main__":
    unittest.main()
