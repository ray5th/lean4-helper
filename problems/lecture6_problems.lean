import Mathlib

-- Lecture 6 Problems

example {a b : ℝ} (h1 : a ^ 2 + b ^ 2 = 0) : a ^ 2 = 0 := by
  apply le_antisymm
  · linarith [sq_nonneg b]
  · exact sq_nonneg a

example {x y : ℝ} (h : x = 1 ∨ y = -1) : x * y + x = y + 1 := by
  obtain hx | hy := h
  · rw [hx]; ring
  · rw [hy]; ring

example {x : ℝ} (hx : 2 * x + 1 = 5) : x = 1 ∨ x = 2 := by
  right
  linarith
