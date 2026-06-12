#!/usr/bin/env python3
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from lean_verifier import LeanEnvironment


def main():
    print("Setting up Lean Environment and pre-caching Mathlib...")
    print("This might take a while if Mathlib has not been built yet.")

    # Initialize the Lean environment with Mathlib
    lean_env = LeanEnvironment(use_mathlib=True)

    print("\nEnvironment initialized successfully!")

    # Run a simple test
    print("\nRunning a quick verification test...")
    test_code = """
import Mathlib

example {x : ℝ} (hx : x ^ 2 - 3 * x + 2 = 0) : x = 1 ∨ x = 2 :=
  sorry
"""
    result = lean_env.verify_proof(test_code)
    print(f"Test Result: {result['status']}")
    if result['status'] != 'success':
        print(f"Errors: {result['errors']}")
        print(f"Goals: {result['goals']}")

if __name__ == "__main__":
    main()
