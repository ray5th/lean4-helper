import unittest
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from lean_verifier import LeanEnvironment

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

if __name__ == '__main__':
    unittest.main()
