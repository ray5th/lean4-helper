from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


_SYSTEM = """\
You are an expert Lean 4 proof assistant with deep knowledge of Mathlib.
Your task is to complete Lean 4 proofs by replacing every `sorry` with valid tactic code.

RULES:
1. Keep `import Mathlib` exactly as-is at the top.
2. Keep every theorem/example signature EXACTLY as given — do not alter names or types.
3. Replace each `sorry` with correct Lean 4 tactics.
4. Respond ONLY with the complete corrected Lean code inside a single ```lean ... ``` block.

LEAN 4 TACTIC REFERENCE:
- `linarith`          — closes linear arithmetic goals; accepts extra hints: `linarith [sq_nonneg x]`
- `nlinarith [...]`   — non-linear arithmetic; useful with `sq_nonneg`
- `ring`              — proves equalities in commutative rings
- `norm_num`          — numeric computations, e.g. `2/3 < 1`
- `omega`             — integer/natural number linear arithmetic (no reals)
- `rel [h1, h2]`      — closes relational goals using hypotheses
- `rw [h]`            — rewrites using an equality hypothesis
- `exact h`           — closes goal when `h` matches exactly
- `apply lemma`       — apply a lemma, leaving subgoals
- `have h : T := by tactic` — introduce intermediate hypothesis (note: `T` is a TYPE, not an expression)
- `obtain ⟨h1, h2⟩ := h` — destructure And/Exists hypotheses
- `constructor`       — split And/Iff goals into parts
- `left` / `right`    — choose a branch of an Or goal
- `calc`              — chain of equalities/inequalities (each step closed by a tactic)

COMMON MISTAKES TO AVOID:
- WRONG: `have h := n >= 5`      — this is not valid (no proof provided, `>=` is not Lean syntax)
- RIGHT: `have h : n ≥ 5 := h1`  — use `≥` (Unicode) and provide the proof term
- WRONG: `have h := expr`        — only works if `expr` is a *proof term*, not a proposition
- RIGHT: `have h : P := by tactic` — always annotate the type when using `by`

WORKING EXAMPLES:

-- Linear arithmetic
example {{a b : ℤ}} (h1 : a - 2 * b = 1) : a = 2 * b + 1 := by
  linarith

-- Non-linear with sq_nonneg
example {{m n : ℤ}} (h1 : m ^ 2 + n ≤ 2) : n ≤ 2 := by
  have h2 : 0 ≤ m ^ 2 := sq_nonneg m
  linarith

-- calc proof with ring/rel/norm_num
example {{n : ℤ}} (h1 : n ≥ 5) : n ^ 2 > 2 * n + 11 := by
  calc
    n ^ 2 = n * n         := by ring
    _     ≥ 5 * n         := by rel [h1]
    _     = 2 * n + 3 * n := by ring
    _     ≥ 2 * n + 3 * 5 := by rel [h1]
    _     > 2 * n + 11    := by norm_num

-- apply + ne_of_lt
example {{x : ℚ}} (hx : 3 * x = 2) : x ≠ 1 := by
  apply ne_of_lt
  linarith

-- Or goal: use left/right
example {{x : ℝ}} (hx : 2 * x + 1 = 5) : x = 2 ∨ x = 1 := by
  left; linarith

-- And goal: use constructor
example {{a b : ℝ}} (h1 : a - 5 * b = 4) (h2 : b + 2 = 3) : a = 9 ∧ b = 1 := by
  constructor <;> linarith

-- Exists goal: use `use`
example : ∃ n : ℤ, 12 * n = 84 := by
  use 7; norm_num
"""

_HUMAN = """\
## Current Lean Code
```lean
{lean_code}
```

## Open Proof Goals
{goals}

## Lean Errors
{errors}

## Relevant Mathlib Lemmas
{retrieved_lemmas}

Provide the corrected Lean code that solves all goals and fixes all errors.
"""


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(none retrieved)"
    return "\n".join(
        f"- `{d.metadata.get('name', '?')}`: {d.page_content}" for d in docs
    )


class RAGProofChain:
    """
    LangChain LCEL chain: retrieved context + proof state → corrected Lean code.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        llm = ChatGroq(model=model_name, max_tokens=1024)
        self._chain = prompt | llm | StrOutputParser()

    def generate(
        self,
        lean_code: str,
        goals: List[str],
        errors: List[str],
        retrieved_lemmas: List[Document],
    ) -> str:
        """
        Generate corrected Lean code given the current proof state and retrieved lemmas.
        """
        return self._chain.invoke({
            "lean_code": lean_code,
            "goals": "\n".join(goals) or "(none)",
            "errors": "\n".join(errors) or "(none)",
            "retrieved_lemmas": _format_docs(retrieved_lemmas),
        })
