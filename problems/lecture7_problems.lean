import Mathlib

-- Lecture 7 Problems

example {x : ℝ} (hx : x ^ 2 - 3 * x + 2 = 0) : x = 1 ∨ x = 2 := by
  have h1 : (x - 1) * (x - 2) = 0 := by linear_combination hx
  have h2 := eq_zero_or_eq_zero_of_mul_eq_zero h1
  obtain hx1 | hx2 := h2
  · left; linarith
  · right; linarith

example {x y : ℤ} (h : 2 * x - y = 4 ∧ y - x + 1 = 2) : x = 5 := by
  obtain ⟨h1, h2⟩ := h
  linarith

example {a b : ℝ} (h1 : a - 5 * b = 4) (h2 : b + 2 = 3) : a = 9 ∧ b = 1 := by
  constructor <;> linarith

example {a : ℚ} (h : ∃ b : ℚ, a = b ^ 2 + 1) : a > 0 := by
  obtain ⟨b, hb⟩ := h
  linarith [sq_nonneg b, hb]

example : ∃ n : ℤ, 12 * n = 84 := by
  use 7
  norm_num
