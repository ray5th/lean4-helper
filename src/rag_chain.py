import subprocess
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class ClaudeCliChat(BaseChatModel):
    """
    LangChain chat model that shells out to the local `claude -p` CLI.

    Useful when a user wants to run the agent on their Claude Pro subscription
    (Pro tokens, no API spend) instead of pay-per-call API keys. The model
    string passed to `claude --model` can be an alias (`opus`, `sonnet`,
    `haiku`) or a full ID (`claude-opus-4-7`).
    """

    model: str = "opus"
    timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Flatten the chat into a single prompt because `claude -p` accepts one
        # string. Tag each segment so the CLI session can see the role split.
        parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(f"<system>\n{m.content}\n</system>")
            elif isinstance(m, HumanMessage):
                parts.append(f"<user>\n{m.content}\n</user>")
            elif isinstance(m, AIMessage):
                parts.append(f"<assistant>\n{m.content}\n</assistant>")
            else:
                parts.append(str(m.content))
        prompt = "\n\n".join(parts)

        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", self.model],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            content = result.stdout.strip()
            if result.returncode != 0 and not content:
                content = f"(claude CLI error: {result.stderr.strip()[:200]})"
        except subprocess.TimeoutExpired:
            content = "(claude CLI timed out)"
        except FileNotFoundError:
            content = "(claude CLI not found on PATH — install Claude Code first)"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


_SYSTEM = """\
You are an expert Lean 4 proof assistant with deep knowledge of Mathlib.
Your task is to complete Lean 4 proofs by replacing every `sorry` with valid tactic code.

RULES:
1. Keep `import Mathlib` exactly as-is at the top.
2. Keep every theorem/example signature EXACTLY as given — do not alter names or types.
3. Replace each `sorry` with correct Lean 4 tactics.
4. Respond ONLY with the complete corrected Lean code inside a single ```lean ... ``` block.
5. Retrieved lemmas are OPTIONAL hints and may be irrelevant — ignore any that
   don't directly help. Never invent lemma names: only cite a retrieved lemma
   or one you are certain exists in Mathlib.
6. Start your code block with a citation comment: `-- used: <FullLemmaName>`
   for each retrieved lemma you actually used, or `-- used: none`.

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

USING A RETRIEVED LEMMA (note the citation comment on the first line):

Given retrieved lemma:
  -- Nat.succ_le_iff
  theorem succ_le_iff {{m n : ℕ}} : succ m ≤ n ↔ m < n

A correct response:
```lean
-- used: Nat.succ_le_iff
import Mathlib

theorem demo {{m n : ℕ}} (h : m < n) : Nat.succ m ≤ n := by
  rw [Nat.succ_le_iff]
  exact h
```
"""

_HUMAN = """\
## Current Lean Code
```lean
{lean_code}
```

## Auto-retrieved Mathlib lemmas — MAY BE IRRELEVANT
These were retrieved automatically and are hints only. Most may not apply.
Use a lemma only if it directly closes or advances a goal; otherwise ignore
them all. Refer to lemmas by the fully-qualified name in the `--` comment
above each declaration.

{retrieved_lemmas}

## Open Proof Goals
{goals}

## Lean Errors
{errors}

Provide the corrected Lean code that solves all goals and fixes all errors.
The FIRST line inside your ```lean block must be a citation comment naming
the retrieved lemma(s) you actually used, or `none`:
`-- used: Nat.add_comm` / `-- used: none`
"""


def _make_llm(model_name: str, api_key: Optional[str]):
    """
    Pick the chat LLM provider from the model name:
      - `claude-cli-*`  → local `claude -p` CLI (uses Pro subscription tokens)
      - `claude-*`      → Anthropic API (requires user-provided api_key)
      - everything else → Groq (api_key optional; falls back to GROQ_API_KEY env)
    """
    if model_name.startswith("claude-cli-"):
        cli_model = model_name[len("claude-cli-"):]
        return ClaudeCliChat(model=cli_model)
    if model_name.startswith("claude-"):
        kwargs = {"model": model_name, "max_tokens": 512}
        if api_key:
            kwargs["anthropic_api_key"] = api_key
        return ChatAnthropic(**kwargs)
    kwargs = {"model": model_name, "max_tokens": 512}
    if api_key:
        kwargs["groq_api_key"] = api_key
    return ChatGroq(**kwargs)


def _format_docs(docs: List[Document]) -> str:
    """
    Render retrieved premises as a fenced Lean block. Each declaration is
    preceded by a `--` comment with its fully-qualified name — page_content
    often shows only the short name (`protected theorem add_comm …`), but the
    model must cite/apply the full name (`Nat.add_comm`).
    """
    if not docs:
        return "(none retrieved)"
    decls = "\n\n".join(
        f"-- {d.metadata.get('name', '?')}\n{d.page_content.strip()}"
        for d in docs
    )
    return f"```lean\n{decls}\n```"


class RAGProofChain:
    """
    LangChain LCEL chain: retrieved context + proof state → corrected Lean code.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        llm = _make_llm(model_name, api_key)
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
