"""
Fuzz tests for the state machine in src/langgraph_agent.py.

Covers:
- should_continue(state): exhaustive combinations of status/attempt/max_retries.
- make_generate_node(chain): behavior with mocked chain output.
- make_verify_node(lean_env): behavior with mocked verifier.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, 'src')

from langgraph.graph import END

import langgraph_agent
from langgraph_agent import (
    make_generate_node,
    make_verify_node,
    should_continue,
)


def _base_state(**overrides):
    """Build a minimal ProofState dict, allowing field overrides."""
    state = {
        "file_path": "/tmp/fake.lean",
        "lean_code": "",
        "goals": [],
        "errors": [],
        "attempt": 0,
        "max_retries": 5,
        "status": "pending",
        "retrieved_lemmas": [],
        "solved_at_attempt": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Fuzz: should_continue
# ---------------------------------------------------------------------------

class TestShouldContinueFuzz(unittest.TestCase):
    """Exhaustive corpus over status x attempt x max_retries."""

    STATUSES = ["pending", "success", "failed"]
    MAX_RETRIES_VALUES = [1, 5, 10]

    def _expected(self, status, attempt, max_retries):
        if status == "success":
            return END
        if attempt >= max_retries:
            return END
        return "retrieve"

    def test_should_continue_fuzz_corpus(self):
        cases = 0
        for status in self.STATUSES:
            for max_retries in self.MAX_RETRIES_VALUES:
                attempts = [0, 1, max_retries - 1, max_retries, max_retries + 1]
                # Drop negative attempts (max_retries=1 -> attempt=0 dup)
                attempts = [a for a in attempts if a >= 0]
                for attempt in attempts:
                    state = _base_state(
                        status=status,
                        attempt=attempt,
                        max_retries=max_retries,
                    )
                    expected = self._expected(status, attempt, max_retries)
                    got = should_continue(state)
                    self.assertEqual(
                        got,
                        expected,
                        msg=(
                            f"should_continue mismatch for "
                            f"status={status!r}, attempt={attempt}, "
                            f"max_retries={max_retries}: "
                            f"expected {expected!r}, got {got!r}"
                        ),
                    )
                    cases += 1
        # Sanity: corpus actually exercised many cases.
        self.assertGreaterEqual(cases, 30)

    def test_success_always_ends(self):
        for attempt in [-1, 0, 1, 5, 100]:
            for max_retries in [1, 5, 10]:
                state = _base_state(
                    status="success", attempt=attempt, max_retries=max_retries
                )
                self.assertEqual(should_continue(state), END)

    def test_attempt_at_max_retries_ends(self):
        for max_retries in [1, 5, 10]:
            state = _base_state(
                status="pending", attempt=max_retries, max_retries=max_retries
            )
            self.assertEqual(should_continue(state), END)

    def test_attempt_exceeds_max_retries_ends(self):
        for max_retries in [1, 5, 10]:
            state = _base_state(
                status="pending", attempt=max_retries + 1, max_retries=max_retries
            )
            self.assertEqual(should_continue(state), END)

    def test_pending_under_max_retries_retrieves(self):
        for max_retries in [1, 5, 10]:
            for attempt in range(max_retries):
                state = _base_state(
                    status="pending", attempt=attempt, max_retries=max_retries
                )
                self.assertEqual(should_continue(state), "retrieve")

    def test_failed_status_also_retries_if_under_max(self):
        """status='failed' is *not* terminal — only success and budget exhaustion are."""
        state = _base_state(status="failed", attempt=0, max_retries=5)
        self.assertEqual(should_continue(state), "retrieve")


# ---------------------------------------------------------------------------
# Fuzz: make_generate_node
# ---------------------------------------------------------------------------

class TestGenerateNodeFuzz(unittest.TestCase):
    def setUp(self):
        # Patch file I/O so generate_node never touches disk.
        self._write_patcher = patch.object(langgraph_agent, "_write_file")
        self._read_patcher = patch.object(langgraph_agent, "_read_file")
        self.mock_write = self._write_patcher.start()
        self.mock_read = self._read_patcher.start()
        self.addCleanup(self._write_patcher.stop)
        self.addCleanup(self._read_patcher.stop)

    def _make_chain(self, output):
        chain = MagicMock()
        chain.generate.return_value = output
        return chain

    # ---- Identical output ----

    def test_chain_returns_identical_code_marks_failed(self):
        """If chain regurgitates the input verbatim, status -> 'failed', attempt += 1, no write."""
        original = "import Mathlib\n\ntheorem foo : True := by trivial"
        chain = self._make_chain(original)
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=2)

        out = node(state)

        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["attempt"], 3)
        self.mock_write.assert_not_called()

    # ---- Fewer theorem blocks ----

    def test_chain_drops_theorem_blocks_is_rejected(self):
        """If the LLM produces fewer theorem-like declarations, reject (no write); still increment attempt."""
        original = (
            "import Mathlib\n\n"
            "theorem foo : True := by sorry\n"
            "theorem bar : True := by sorry\n"
        )
        # Output drops 'bar' entirely.
        bad_output = (
            "```lean\n"
            "import Mathlib\n\n"
            "theorem foo : True := by trivial\n"
            "```"
        )
        chain = self._make_chain(bad_output)
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=1)

        out = node(state)

        self.assertEqual(out["attempt"], 2)
        self.mock_write.assert_not_called()
        # lean_code must not have been replaced with the dropped version.
        self.assertEqual(out["lean_code"], original)

    # ---- More theorem blocks ----

    def test_chain_adds_theorem_blocks_is_accepted(self):
        """More theorem blocks than original — still accepted (count check is one-sided)."""
        original = (
            "import Mathlib\n\n"
            "theorem foo : True := by sorry\n"
        )
        new_output = (
            "```lean\n"
            "import Mathlib\n\n"
            "theorem foo : True := by trivial\n"
            "theorem helper : True := by trivial\n"
            "```"
        )
        chain = self._make_chain(new_output)
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=0)

        out = node(state)

        self.assertEqual(out["attempt"], 1)
        self.mock_write.assert_called_once()
        # lean_code should be updated.
        self.assertIn("theorem helper", out["lean_code"])

    # ---- Empty string ----

    def test_chain_returns_empty_string_marks_failed(self):
        """Empty output from the chain must terminate with status='failed', not silently write garbage."""
        original = "import Mathlib\n\ntheorem foo : True := by sorry"
        chain = self._make_chain("")
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=0)

        out = node(state)

        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["attempt"], 1)
        self.mock_write.assert_not_called()

    def test_chain_returns_whitespace_only_marks_failed(self):
        """Whitespace-only output also counts as nothing useful."""
        original = "import Mathlib\n\ntheorem foo : True := by sorry"
        chain = self._make_chain("   \n\t  \n")
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=0)

        out = node(state)

        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["attempt"], 1)
        self.mock_write.assert_not_called()

    # ---- Proper theorem block ----

    def test_chain_returns_valid_proof_writes_file(self):
        """A well-formed replacement gets written, attempt increments, no 'failed'."""
        original = "import Mathlib\n\ntheorem foo : True := by sorry"
        good_output = (
            "```lean\n"
            "import Mathlib\n\n"
            "theorem foo : True := by trivial\n"
            "```"
        )
        chain = self._make_chain(good_output)
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=2, file_path="/tmp/x.lean")

        out = node(state)

        self.assertEqual(out["attempt"], 3)
        self.assertNotEqual(out.get("status"), "failed")
        self.mock_write.assert_called_once()
        # Path must be passed through.
        called_path, called_code = self.mock_write.call_args[0]
        self.assertEqual(called_path, "/tmp/x.lean")
        self.assertIn("trivial", called_code)
        # State's lean_code is updated to the new code.
        self.assertEqual(out["lean_code"], called_code)

    def test_chain_output_with_no_code_fences_still_works(self):
        """Raw lean code (no ```lean fence) should be accepted."""
        original = "import Mathlib\n\ntheorem foo : True := by sorry"
        raw = "theorem foo : True := by trivial"
        chain = self._make_chain(raw)
        node = make_generate_node(chain)
        state = _base_state(lean_code=original, attempt=0)

        out = node(state)

        self.assertEqual(out["attempt"], 1)
        self.mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Fuzz: make_verify_node
# ---------------------------------------------------------------------------

class TestVerifyNodeFuzz(unittest.TestCase):
    def setUp(self):
        self._read_patcher = patch.object(langgraph_agent, "_read_file")
        self.mock_read = self._read_patcher.start()
        self.addCleanup(self._read_patcher.stop)
        # Default file contents from disk
        self.mock_read.return_value = "import Mathlib\n\ntheorem foo : True := by trivial"

    def _make_env(self, status, errors, goals):
        env = MagicMock()
        env.verify_proof.return_value = {
            "status": status,
            "errors": errors,
            "goals": goals,
        }
        return env

    def test_success_marks_state_success_and_solved_at_attempt(self):
        env = self._make_env("success", [], [])
        node = make_verify_node(env)
        # attempt=2 means the *current* attempt counter is 2; the run we are
        # about to perform is attempt #3 -> solved_at_attempt should be 3.
        state = _base_state(attempt=2, solved_at_attempt=0)

        out = node(state)

        self.assertEqual(out["status"], "success")
        self.assertEqual(out["solved_at_attempt"], 3)
        self.assertEqual(out["errors"], [])
        self.assertEqual(out["goals"], [])

    def test_failure_marks_state_pending_and_keeps_errors_goals(self):
        env = self._make_env("failure", ["err1"], ["goal1"])
        node = make_verify_node(env)
        state = _base_state(attempt=0, solved_at_attempt=0)

        out = node(state)

        self.assertEqual(out["status"], "pending")
        self.assertEqual(out["errors"], ["err1"])
        self.assertEqual(out["goals"], ["goal1"])
        # solved_at_attempt stays unchanged on failure.
        self.assertEqual(out["solved_at_attempt"], 0)

    def test_failure_then_success_records_correct_solved_attempt(self):
        """A success on the 5th attempt should record solved_at_attempt=5."""
        env_fail = self._make_env("failure", ["err"], ["goal"])
        env_ok = self._make_env("success", [], [])

        state = _base_state(attempt=0, solved_at_attempt=0)
        # Simulate four failures then a success.
        for i in range(4):
            node = make_verify_node(env_fail)
            state = node(state)
            self.assertEqual(state["status"], "pending")
            # The agent's generate-step would normally bump 'attempt'; simulate that.
            state["attempt"] += 1

        # Now: attempt=4, next verify should be attempt #5.
        node_ok = make_verify_node(env_ok)
        final = node_ok(state)
        self.assertEqual(final["status"], "success")
        self.assertEqual(final["solved_at_attempt"], 5)

    def test_verify_reads_from_disk(self):
        env = self._make_env("success", [], [])
        node = make_verify_node(env)
        state = _base_state(file_path="/tmp/specific.lean")

        out = node(state)

        self.mock_read.assert_called_once_with("/tmp/specific.lean")
        # lean_code in state should be what was read.
        self.assertEqual(out["lean_code"], self.mock_read.return_value)


if __name__ == "__main__":
    unittest.main()
