import Mathlib

-- Lecture 4 Problems

example {n : ℤ} (h1 : n ≥ 5) : n^2 > 2 * n + 11 := by
  calc
    n^2 = n * n         := by ring
    _   ≥ 5 * n         := by rel [h1]
    _   = 2 * n + 3 * n := by ring
    _   ≥ 2 * n + 3 * 5 := by rel [h1]
    _   > 2 * n + 11    := by norm_num

example {x y : ℝ} (h : x ^ 2 + y ^ 2 ≤ 1) : (x + y) ^ 2 < 3 := by
  calc
    (x + y) ^ 2 ≤ (x + y) ^ 2 + (x - y) ^ 2 := by linarith [sq_nonneg (x - y)]
    _           = 2 * (x ^ 2 + y ^ 2)        := by ring
    _           ≤ 2 * 1                       := by rel [h]
    _           < 3                           := by norm_num

example {m n : ℤ} (h1 : m + 3 ≤ 2 * n - 1) (h2 : n ≤ 5) : m ≤ 6 := by
  linarith
