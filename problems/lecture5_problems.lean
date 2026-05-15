import Mathlib

-- Lecture 5 Problems

example {r s : ℚ} (h1 : s + 3 ≥ r) (h2 : s + r ≤ 3) : r ≤ 3 := by
  linarith

example {x : ℚ} (hx : 3 * x = 2) : x ≠ 1 := by
  apply ne_of_lt
  linarith

example {y : ℝ} : y ^ 2 + 1 ≠ 0 := by
  apply ne_of_gt
  linarith [sq_nonneg y]
