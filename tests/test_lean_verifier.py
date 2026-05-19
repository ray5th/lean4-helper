import unittest
import sys
import os
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, 'src')

try:
    from lean_interact import LeanServer
    LEAN_INTERACT_AVAILABLE = True
except ImportError:
    LEAN_INTERACT_AVAILABLE = False

from lean_verifier import LeanEnvironment


@unittest.skipUnless(LEAN_INTERACT_AVAILABLE, "lean-interact not installed")
class TestLeanVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We'll use Mathlib here since our goal is to verify it works with the MVP setup
        cls.lean_env = LeanEnvironment(use_mathlib=True)

    def test_correct_proof(self):
        lean_code = """
import Mathlib

theorem add_comm_test (n m : Nat) : n + m = m + n := by
  exact Nat.add_comm n m
"""
        result = self.lean_env.verify_proof(lean_code)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["errors"]), 0)
        self.assertEqual(len(result["goals"]), 0)

    def test_incorrect_proof_type_mismatch(self):
        lean_code = """
import Mathlib

theorem add_comm_test (n m : Nat) : n + m = m + n := by
  exact n
"""
        result = self.lean_env.verify_proof(lean_code)
        self.assertEqual(result["status"], "failure")
        self.assertTrue(any("type mismatch" in err or "application type mismatch" in err for err in result["errors"]),
                        f"Expected type mismatch error, got: {result['errors']}")

    def test_incomplete_proof_sorry(self):
        lean_code = """
import Mathlib

theorem my_incomplete_thm (n : Nat) : n = 5 → n = 5 := by
  sorry
"""
        result = self.lean_env.verify_proof(lean_code)
        self.assertEqual(result["status"], "failure")
        # Ensure it has an error indicating sorry
        self.assertTrue(any("uses 'sorry'" in err for err in result["errors"]),
                        f"Expected sorry warning/error, got: {result['errors']}")
        # Ensure it outputs the goal
        self.assertEqual(len(result["goals"]), 1)
        self.assertIn("⊢ n = 5 → n = 5", result["goals"][0])


def _make_response(messages=None, sorries=None, env=None):
    """Build a mock response object that mimics the lean_interact CommandResponse shape.

    messages: list of dicts {severity, data}
    sorries: list of dicts {goal}
    """
    msg_objs = []
    for m in (messages or []):
        msg_objs.append(SimpleNamespace(severity=m.get("severity"), data=m.get("data")))
    sorry_objs = []
    for s in (sorries or []):
        sorry_objs.append(SimpleNamespace(goal=s.get("goal")))
    return SimpleNamespace(messages=msg_objs, sorries=sorry_objs, env=env)


class TestLeanVerifierMocked(unittest.TestCase):
    """Edge-case coverage for LeanEnvironment.verify_proof using a mocked LeanServer.

    The mock patches `lean_verifier.LeanServer` BEFORE the LeanEnvironment is
    constructed so we never need a real Lean installation. Each test configures
    the mock's `run()` to return a canned response (or raise) for the input case.
    """

    def setUp(self):
        # Patch LeanServer and LeanREPLConfig / TempRequireProject so __init__ is cheap.
        patcher_server = patch('lean_verifier.LeanServer')
        patcher_config = patch('lean_verifier.LeanREPLConfig')
        patcher_proj = patch('lean_verifier.TempRequireProject')

        self.MockServer = patcher_server.start()
        self.MockConfig = patcher_config.start()
        self.MockProject = patcher_proj.start()

        self.addCleanup(patcher_server.stop)
        self.addCleanup(patcher_config.stop)
        self.addCleanup(patcher_proj.stop)

        # Instance returned by LeanServer(config)
        self.mock_server_instance = MagicMock()
        self.MockServer.return_value = self.mock_server_instance

        # Construct env AFTER patches are active so the patched LeanServer is used.
        self.env = LeanEnvironment(use_mathlib=True)

    def test_empty_string_does_not_crash(self):
        """Empty input must not crash. Should either skip the server call or
        return a clean failure with a descriptive error."""
        # If the verifier short-circuits, server.run won't be called.
        # If it does call, simulate a successful empty response.
        self.mock_server_instance.run.return_value = _make_response()

        result = self.env.verify_proof("")
        self.assertEqual(result["status"], "failure")
        self.assertTrue(len(result["errors"]) > 0,
                        f"Expected an error message for empty input, got: {result}")
        self.assertEqual(result["goals"], [])

    def test_whitespace_only_does_not_crash(self):
        """Whitespace-only input behaves the same as empty: clean failure, no crash."""
        self.mock_server_instance.run.return_value = _make_response()

        result = self.env.verify_proof("   \n  ")
        self.assertEqual(result["status"], "failure")
        self.assertTrue(len(result["errors"]) > 0,
                        f"Expected an error message for whitespace input, got: {result}")
        self.assertEqual(result["goals"], [])

    def test_malformed_lean_returns_failure_with_error(self):
        """Mock returns an error message; verify_proof should surface it."""
        self.mock_server_instance.run.return_value = _make_response(
            messages=[{"severity": "error", "data": "syntax error"}],
            sorries=[],
        )

        result = self.env.verify_proof("theorem foo := this is garbage")
        self.assertEqual(result["status"], "failure")
        self.assertIn("syntax error", result["errors"])
        self.assertEqual(result["goals"], [])

    def test_unicode_heavy_input_returns_sorry_goal(self):
        """Unicode-heavy Lean input with a sorry should produce a goal."""
        goal_text = "n m : ℕ\n⊢ ∀ n : ℕ, ∃ m, n + m = m + n"
        self.mock_server_instance.run.return_value = _make_response(
            messages=[{"severity": "warning", "data": "declaration uses 'sorry'"}],
            sorries=[{"goal": goal_text}],
        )

        result = self.env.verify_proof(
            "theorem foo : ∀ n : ℕ, ∃ m, n + m = m + n := by sorry"
        )
        self.assertEqual(result["status"], "failure")
        self.assertTrue(any("sorry" in e for e in result["errors"]))
        self.assertEqual(len(result["goals"]), 1)
        self.assertEqual(result["goals"][0], goal_text)

    def test_very_long_input_does_not_crash(self):
        """A very large input must not blow up the verifier."""
        long_code = "foo " * 10000  # ~40k chars
        self.mock_server_instance.run.return_value = _make_response(
            messages=[{"severity": "error", "data": "syntax error"}],
            sorries=[],
        )

        # Should simply pass through to server.run (verifier itself must not
        # do quadratic-cost work on the input string).
        result = self.env.verify_proof(long_code)
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["errors"], ["syntax error"])

    def test_server_run_raises_is_caught(self):
        """If LeanServer.run raises, verify_proof must catch it and return a
        failure result rather than propagating the exception."""
        self.mock_server_instance.run.side_effect = Exception("connection lost")

        result = self.env.verify_proof("theorem foo : True := trivial")
        self.assertEqual(result["status"], "failure")
        self.assertTrue(len(result["errors"]) > 0)
        self.assertTrue(any("connection lost" in e for e in result["errors"]),
                        f"Expected exception message in errors, got: {result['errors']}")
        self.assertEqual(result["goals"], [])

    def test_empty_response_object_does_not_crash(self):
        """If the server returns an object without messages/sorries attrs,
        verify_proof must handle it cleanly."""
        # Use an object that has no messages / sorries attributes at all.
        bare = SimpleNamespace()
        self.mock_server_instance.run.return_value = bare

        result = self.env.verify_proof("theorem foo : True := trivial")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["goals"], [])

    def test_successful_response_returns_success(self):
        """Empty messages and sorries -> status success."""
        self.mock_server_instance.run.return_value = _make_response(
            messages=[], sorries=[], env=42,
        )

        result = self.env.verify_proof("theorem foo : True := trivial")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["goals"], [])
        self.assertEqual(result["env"], 42)


if __name__ == '__main__':
    unittest.main()
